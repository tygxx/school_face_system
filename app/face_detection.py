import cv2
import face_recognition
import numpy as np
from typing import List, Tuple, Dict
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import os
import platform

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
        # 添加人脸验证历史记录
        self.face_verification_history = {}
        # 配置参数
        self.REQUIRED_CONSECUTIVE_MATCHES = 3  # 需要连续匹配的次数
        self.CONFIDENCE_THRESHOLD = 0.5  # 置信度阈值, 0.5表示50%的置信度, 该值越小，越严格 
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
                    face_encoding = np.frombuffer(face_encoding_bytes, dtype=np.float64)
                    logger.debug(f"学生 {student['name']} 的人脸编码形状: {face_encoding.shape}")
                    logger.debug(f"人脸编码数据类型: {face_encoding.dtype}")
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_metadata.append({
                        'type': 'student',
                        'id': student_id,
                        'name': student['name'],
                        'class_name': student['class_name']
                    })
            
            # 加载监护人人脸数据
            for guardian_id, face_encoding_bytes in face_data['guardians'].items():
                guardian = self.db.get_guardian_by_id(guardian_id)
                if guardian and face_encoding_bytes:
                    face_encoding = np.frombuffer(face_encoding_bytes, dtype=np.float64)
                    logger.info(f"监护人 {guardian['name']} 的人脸编码形状: {face_encoding.shape}")
                    logger.info(f"人脸编码数据类型: {face_encoding.dtype}")
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_metadata.append({
                        'type': 'guardian',
                        'id': guardian_id,
                        'name': guardian['name'],
                        'phone': guardian['phone']
                    })
            
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
            # 缩小图像以加快处理速度
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # 检测人脸位置
            face_locations = self.detect_faces(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_results = []
            current_timestamp = datetime.now().timestamp()
            
            for face_encoding, face_location in zip(face_encodings, face_locations):
                # 在已知人脸中查找匹配
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                
                best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None
                
                # 添加调试信息
                if len(face_distances) > 0:
                    logger.debug(f"人脸距离: {face_distances}")
                    logger.debug(f"最佳匹配索引: {best_match_index}")
                    logger.debug(f"最佳匹配距离: {face_distances[best_match_index]}")
                    logger.debug(f"匹配结果: {matches}")

                # 获取当前检测到的人脸的特征向量字符串表示，用作唯一标识
                face_id = str(face_encoding.tobytes())
                
                if best_match_index is not None and matches[best_match_index]:
                    metadata = self.known_face_metadata[best_match_index].copy()
                    confidence = 1 - face_distances[best_match_index]
                    
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
                    
                    # 验证连续匹配和置信度
                    if len(recent_matches) >= self.REQUIRED_CONSECUTIVE_MATCHES:
                        # 检查是否所有最近的匹配都指向同一个人
                        same_person = all(
                            match['metadata']['id'] == metadata['id'] 
                            and match['confidence'] >= self.CONFIDENCE_THRESHOLD
                            for match in recent_matches
                        )
                        
                        if not same_person:
                            metadata = {"type": "unknown"}
                            logger.debug(f"人脸验证失败：连续匹配不一致或置信度不足")
                else:
                    metadata = {"type": "unknown"}
                
                # 清理过期的历史记录
                self._cleanup_verification_history(current_timestamp)
                
                # 转换回原始图像大小
                top, right, bottom, left = face_location
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
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
                # 创建一个空白的PIL图像用于绘制文字
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
                    # 获取文本边界框
                    bbox = draw.textbbox((left, y), line, font=font)
                    # 绘制白色背景
                    draw.rectangle([(bbox[0], bbox[1] - 5), (bbox[2], bbox[3] + 5)], fill=(255, 255, 255))
                    # 绘制文字
                    draw.text((left, y), line, font=font, fill=color[::-1])  # PIL使用RGB而不是BGR
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
        try:
            # 先进行图像预处理
            processed_frame = self.preprocess_frame(frame)
            
            # 尝试使用CNN模型
            face_locations = face_recognition.face_locations(
                processed_frame,
                model="cnn",
                number_of_times_to_upsample=1
            )
        except Exception as e:
            # 如果CNN失败，回退到HOG模型
            print("GPU模型加载失败，使用CPU模型")
            face_locations = face_recognition.face_locations(
                processed_frame,
                model="hog",  # 使用CPU友好的HOG模型
                number_of_times_to_upsample=1
            )
        
        return face_locations

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

    def preprocess_frame(self, frame): # 图像预处理，调整大小和光线补偿
        # 调整大小
        height, width = frame.shape[:2]
        if width > 640:  # 保持合适的处理大小
            frame = cv2.resize(frame, (640, int(height * 640 / width)))
        
        # 光线补偿
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return frame

    def _cleanup_verification_history(self, current_timestamp, max_age=30): # 自动清理30秒内未出现的人脸记录，防止内存占用过大
        """清理超过指定时间的人脸验证历史记录"""
        expired_faces = []
        for face_id, history in self.face_verification_history.items():
            if current_timestamp - history['last_seen'] > max_age:
                expired_faces.append(face_id)
        
        for face_id in expired_faces:
            del self.face_verification_history[face_id]