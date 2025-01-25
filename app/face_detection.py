import cv2
import numpy as np
from typing import List, Tuple, Dict
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import os
import platform
import insightface
from insightface.app import FaceAnalysis

from .database import DatabaseManager
from .utils.logger import setup_logger, handle_exceptions
from .utils.exceptions import CameraError, FaceDetectionError

# 创建模块日志记录器
logger = setup_logger('face_detection')

def get_system_font():
    """获取系统对应的中文字体路径"""
    system = platform.system()
    if system == 'Darwin':  # macOS
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Arial Unicode.ttf"
        ]
    elif system == 'Windows':
        font_paths = [
            "C:\\Windows\\Fonts\\msyh.ttc",  # 微软雅黑
            "C:\\Windows\\Fonts\\simsun.ttc",  # 宋体
            "C:\\Windows\\Fonts\\simhei.ttf"   # 黑体
        ]
    else:  # Linux 和其他系统
        font_paths = [
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
        ]

    # 尝试查找可用的字体
    for font_path in font_paths:
        if os.path.exists(font_path):
            return font_path
            
    return None

class FaceDetector:
    def __init__(self):
        self.db = DatabaseManager()
        self.known_face_encodings = []
        self.known_face_metadata = []
        self.system_font_path = get_system_font()
        
        # 初始化 InsightFace
        self.face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("InsightFace model initialized")
        
        # 特征维度
        self.FEATURE_DIM = 512  # InsightFace 特征维度
        
        # 添加人脸验证历史记录
        self.face_verification_history = {}
        # 配置参数
        self.REQUIRED_CONSECUTIVE_MATCHES = 3  # 需要连续匹配的次数
        self.CONFIDENCE_THRESHOLD = 0.3  # 置信度阈值，由于使用余弦相似度，调整为0.3
        self.MAX_HISTORY_SIZE = 10  # 历史记录最大保存帧数
        self.load_registered_faces()
        logger.info("FaceDetector initialized")
        
    def load_registered_faces(self):
        """从数据库加载所有已注册的人脸数据"""
        try:
            face_data = self.db.get_all_face_encodings()
            
            # 加载学生人脸数据
            for student_id, face_encoding_bytes in face_data['students'].items():
                student = self.db.get_student_by_id(student_id)
                if student and face_encoding_bytes:
                    try:
                        face_encoding = np.frombuffer(face_encoding_bytes, dtype=np.float32)
                        # 检查特征维度
                        if len(face_encoding) != self.FEATURE_DIM:
                            logger.warning(f"跳过维度不匹配的特征: {student['name']}, 维度: {len(face_encoding)}")
                            continue
                        logger.debug(f"学生 {student['name']} 的人脸编码形状: {face_encoding.shape}")
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_metadata.append({
                            'type': 'student',
                            'id': student_id,
                            'name': student['name'],
                            'class_name': student['class_name']
                        })
                    except Exception as e:
                        logger.error(f"加载学生 {student['name']} 的人脸特征失败: {str(e)}")
            
            # 加载监护人人脸数据
            for guardian_id, face_encoding_bytes in face_data['guardians'].items():
                guardian = self.db.get_guardian_by_id(guardian_id)
                if guardian and face_encoding_bytes:
                    try:
                        face_encoding = np.frombuffer(face_encoding_bytes, dtype=np.float32)
                        # 检查特征维度
                        if len(face_encoding) != self.FEATURE_DIM:
                            logger.warning(f"跳过维度不匹配的特征: {guardian['name']}, 维度: {len(face_encoding)}")
                            continue
                        logger.info(f"监护人 {guardian['name']} 的人脸编码形状: {face_encoding.shape}")
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_metadata.append({
                            'type': 'guardian',
                            'id': guardian_id,
                            'name': guardian['name'],
                            'phone': guardian['phone']
                        })
                    except Exception as e:
                        logger.error(f"加载监护人 {guardian['name']} 的人脸特征失败: {str(e)}")
            
            logger.info(f"已加载 {len(self.known_face_encodings)} 个人脸特征")
            
            # 添加调试信息：检查人脸比对过程
            if len(self.known_face_encodings) > 0:
                logger.debug(f"第一个人脸编码示例：{self.known_face_encodings[0][:5]}")  # 只显示前5个值
        except Exception as e:
            logger.error(f"加载人脸数据失败: {str(e)}")
            raise FaceDetectionError("Failed to load face data")

    @handle_exceptions(logger)
    def add_face_encoding(self, face_encoding: np.ndarray, metadata: Dict):
        """
        添加已知人脸编码和相关信息到系统
        """
        try:
            self.known_face_encodings.append(face_encoding)
            self.known_face_metadata.append(metadata)
            logger.debug(f"Added new face encoding with metadata: {metadata}")
        except Exception as e:
            raise FaceDetectionError(f"Failed to add face encoding: {str(e)}")
        
    @handle_exceptions(logger)
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        处理视频帧，检测和识别人脸
        """
        if frame is None:
            raise FaceDetectionError("Invalid frame received")
            
        try:
            # 检测人脸和提取特征
            face_results = []
            current_timestamp = datetime.now().timestamp()
            
            # 获取人脸位置和特征
            faces = self.detect_faces(frame)
            
            for (bbox, face_embedding) in faces:
                # 在已知人脸中查找匹配
                if len(self.known_face_encodings) > 0:
                    # 计算与所有已知人脸的相似度
                    similarities = [
                        np.dot(face_embedding, known_encoding) / 
                        (np.linalg.norm(face_embedding) * np.linalg.norm(known_encoding)) 
                        for known_encoding in self.known_face_encodings
                    ]
                    best_match_index = np.argmax(similarities)
                    similarity = similarities[best_match_index]
                    
                    # 添加调试信息
                    logger.debug(f"人脸相似度: {similarities}")
                    logger.debug(f"最佳匹配索引: {best_match_index}")
                    logger.debug(f"最佳匹配相似度: {similarity}")

                    # 获取当前检测到的人脸的特征向量字符串表示，用作唯一标识
                    face_id = str(face_embedding.tobytes())
                    
                    # 更严格的匹配逻辑
                    if similarity > 0.5:  # 可以根据需要调整阈值
                        metadata = self.known_face_metadata[best_match_index].copy()
                        confidence = similarity
                        
                        # 更新人脸验证历史
                        if face_id not in self.face_verification_history:
                            self.face_verification_history[face_id] = {
                                'matches': [],
                                'last_seen': current_timestamp
                            }
                        
                        # 添加新的匹配结果
                        self.face_verification_history[face_id]['matches'].append({
                            'metadata': metadata,
                            'confidence': confidence,
                            'timestamp': current_timestamp
                        })
                        
                        # 保持历史记录在最大大小以内
                        if len(self.face_verification_history[face_id]['matches']) > self.MAX_HISTORY_SIZE:
                            self.face_verification_history[face_id]['matches'].pop(0)
                        
                        # 检查最近的匹配结果
                        recent_matches = self.face_verification_history[face_id]['matches'][-self.REQUIRED_CONSECUTIVE_MATCHES:]
                        
                        # 更严格的验证逻辑：要求连续匹配且平均置信度达到阈值
                        if len(recent_matches) >= self.REQUIRED_CONSECUTIVE_MATCHES:
                            same_person = all(
                                match['metadata']['id'] == metadata['id']
                                for match in recent_matches
                            )
                            avg_confidence = sum(match['confidence'] for match in recent_matches) / len(recent_matches)
                            
                            if not same_person or avg_confidence < self.CONFIDENCE_THRESHOLD:
                                metadata = {"type": "unknown"}
                                logger.debug(f"人脸验证失败：连续匹配不一致或平均置信度不足 ({avg_confidence:.3f})")
                    else:
                        metadata = {"type": "unknown"}
                        logger.debug(f"人脸验证失败：相似度 {similarity:.3f} 低于阈值")
                else:
                    metadata = {"type": "unknown"}
                
                # 清理过期的历史记录
                self._cleanup_verification_history(current_timestamp)
                
                # 获取边界框坐标
                top, left, bottom, right = bbox[1], bbox[0], bbox[3], bbox[2]
                
                # 在图像上绘制方框和标签
                if metadata["type"] == "unknown":
                    color = (0, 0, 255)  # 红色
                    label = "未知人员"
                elif metadata["type"] == "student":
                    color = (0, 255, 0)  # 绿色
                    label = f"学生: {metadata['name']}\n班级: {metadata['class_name']}"
                else:  # guardian
                    color = (255, 0, 0)  # 蓝色
                    label = f"监护人: {metadata['name']}"
                
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # 使用支持中文的方式显示标签
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                
                # 使用系统字体
                font_size = 24
                if self.system_font_path:
                    try:
                        font = ImageFont.truetype(self.system_font_path, font_size)
                    except OSError:
                        logger.warning(f"无法加载系统字体: {self.system_font_path}")
                        font = ImageFont.load_default()
                else:
                    logger.warning("未找到合适的系统字体，使用默认字体")
                    font = ImageFont.load_default()

                # 计算文本位置
                y = top - 30
                for line in label.split('\n'):
                    bbox = draw.textbbox((left, y), line, font=font)
                    draw.rectangle([(bbox[0], bbox[1] - 5), (bbox[2], bbox[3] + 5)], fill=(255, 255, 255))
                    draw.text((left, y), line, font=font, fill=color[::-1])
                    y += (bbox[3] - bbox[1]) + 5

                # 转换回OpenCV格式
                frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                face_results.append({
                    "location": (top, right, bottom, left),
                    "metadata": metadata
                })
            
            return frame, face_results
            
        except Exception as e:
            raise FaceDetectionError(f"Error processing frame: {str(e)}")

    def detect_faces(self, frame):
        """使用 InsightFace 检测人脸"""
        try:
            # 使用 InsightFace 进行人脸检测和特征提取
            faces = self.face_app.get(frame)
            return [(face.bbox.astype(int), face.embedding) for face in faces]
        except Exception as e:
            logger.error(f"人脸检测失败: {str(e)}")
            return []

    def verify_relationship(self, student_id: str, guardian_id: str) -> bool:
        """验证学生和监护人的关系"""
        return self.db.verify_relationship(student_id, guardian_id)

    def check_pickup_authorization(self, face_results: List[Dict]) -> Tuple[bool, str]:
        """检查接送授权"""
        students = [r for r in face_results if r["metadata"].get("type") == "student"]
        guardians = [r for r in face_results if r["metadata"].get("type") == "guardian"]
        
        if not students or not guardians:
            return True, ""  # 没有同时检测到学生和监护人，不需要验证
            
        for student in students:
            student_id = student["metadata"]["id"]
            student_name = student["metadata"]["name"]
            
            for guardian in guardians:
                guardian_id = guardian["metadata"]["id"]
                guardian_name = guardian["metadata"]["name"]
                
                if not self.verify_relationship(student_id, guardian_id):
                    msg = f"⚠️ 警告：{guardian_name} 不是 {student_name} 的授权监护人！"
                    logger.warning(msg)
                    return False, msg
        
        return True, ""

    @handle_exceptions(logger)
    def start_video_detection(self, camera_id: int = 0):
        """
        启动视频检测
        """
        logger.info(f"Starting video detection with camera ID: {camera_id}")
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise CameraError(f"Failed to open camera {camera_id}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    raise CameraError("Failed to capture frame from camera")
                
                processed_frame, face_results = self.process_frame(frame)
                
                # 检查接送授权
                authorized, message = self.check_pickup_authorization(face_results)
                if not authorized:
                    # 使用PIL绘制警告消息
                    pil_img = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)
                    
                    # 使用系统字体
                    font_size = 32  # 警告文字稍大一些
                    if self.system_font_path:
                        try:
                            font = ImageFont.truetype(self.system_font_path, font_size)
                        except OSError:
                            logger.warning(f"无法加载系统字体: {self.system_font_path}")
                            font = ImageFont.load_default()
                    else:
                        logger.warning("未找到合适的系统字体，使用默认字体")
                        font = ImageFont.load_default()
                    
                    # 获取文本边界框
                    bbox = draw.textbbox((10, 30), message, font=font)
                    # 绘制半透明黑色背景
                    background_img = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
                    background_draw = ImageDraw.Draw(background_img)
                    background_draw.rectangle(
                        [(bbox[0] - 5, bbox[1] - 5), (bbox[2] + 5, bbox[3] + 5)],
                        fill=(0, 0, 0, 128)
                    )
                    # 将半透明背景叠加到原图
                    pil_img = Image.alpha_composite(pil_img.convert('RGBA'), background_img)
                    # 绘制警告文字
                    draw = ImageDraw.Draw(pil_img)
                    draw.text((10, 30), message, font=font, fill=(255, 0, 0))  # 红色文字
                    
                    # 转换回OpenCV格式
                    processed_frame = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
                
                # 显示处理后的画面
                cv2.imshow('Face Detection', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested to quit")
                    break
                    
        except Exception as e:
            logger.error(f"Error in video detection: {str(e)}", exc_info=True)
            raise
        finally:
            logger.info("Cleaning up resources")
            cap.release()
            cv2.destroyAllWindows() 

    def preprocess_frame(self, frame):  # 图像预处理，调整大小和光线补偿
        # 调整大小
        height, width = frame.shape[:2]
        if width > 640:  # 保持合适的处理大小
            frame = cv2.resize(frame, (640, int(height * 640 / width)))
        
        # 光线补偿
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge((l_channel, a_channel, b_channel))
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return frame

    def _cleanup_verification_history(self, current_timestamp, max_age=30):  # 自动清理30秒内未出现的人脸记录
        """清理超过指定时间的人脸验证历史记录"""
        expired_faces = []
        for face_id, history in self.face_verification_history.items():
            if current_timestamp - history['last_seen'] > max_age:
                expired_faces.append(face_id)
        
        for face_id in expired_faces:
            del self.face_verification_history[face_id]