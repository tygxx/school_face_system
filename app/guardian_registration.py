import cv2
import numpy as np
from typing import Optional, Tuple
from .database import DatabaseManager
from .utils.logger import setup_logger
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw, ImageFont
import os

logger = setup_logger('guardian_registration')


class GuardianRegistration:
    def __init__(self):
        self.db = DatabaseManager()
        logger.info("正在加载人脸识别模型...")
        print("正在加载人脸识别模型，请稍候...")
        # 初始化 InsightFace
        self.face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("InsightFace model initialized")

    def capture_face(self) -> Tuple[bool, Optional[np.ndarray], Optional[str]]:
        """
        采集人脸图像
        返回：(是否成功, 人脸编码数据, 错误信息)
        """
        logger.info("正在初始化摄像头...")
        print("正在打开摄像头...")
        
        # 尝试多个摄像头索引
        cap = None
        for camera_index in [0, 1]:
            try:
                cap = cv2.VideoCapture(camera_index)
                if cap.isOpened():
                    logger.info(f"成功打开摄像头 {camera_index}")
                    break
            except Exception as e:
                logger.warning(f"尝试打开摄像头 {camera_index} 失败: {str(e)}")
                if cap is not None:
                    cap.release()

        if cap is None or not cap.isOpened():
            logger.error("无法打开任何摄像头")
            return False, None, "无法打开摄像头"

        # 设置摄像头参数并验证
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 验证摄像头设置是否成功
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.info(f"摄像头分辨率: {actual_width}x{actual_height}")

        # 预热摄像头
        logger.info("预热摄像头...")
        for _ in range(10):
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("预热过程中无法读取画面，重试...")
                continue
            cv2.waitKey(100)  # 等待100ms

        # 创建并设置窗口
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Original', 640, 480)
        cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Face Detection', 640, 480)

        try:
            logger.info("开始捕获视频流...")
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.error("无法读取摄像头画面")
                    return False, None, "无法读取摄像头画面"

                frame = cv2.resize(frame, (640, 480))
                cv2.imshow('Original', frame)
                
                # 使用 InsightFace 检测人脸
                faces = self.face_app.get(frame)
                
                # 在画面中标记人脸
                frame_draw = frame.copy()
                for face in faces:
                    bbox = face.bbox.astype(int)
                    cv2.rectangle(
                        frame_draw,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        (0, 255, 0),  # 绿色
                        2
                    )

                cv2.imshow('Face Detection', frame_draw)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("用户取消采集")
                    return False, None, "用户取消"
                elif key == ord(' '):  # 按空格键拍照
                    # 检测到人脸才能拍照
                    if len(faces) == 1:
                        # 获取人脸特征
                        face_embedding = faces[0].embedding
                        if face_embedding is not None:
                            logger.info("成功捕获人脸特征")
                            return True, face_embedding, None
                        else:
                            logger.warning("无法提取人脸特征")
                            return False, None, "无法提取人脸特征"
                    elif len(faces) == 0:
                        logger.warning("未检测到人脸，请正对摄像头")
                    else:
                        logger.warning(f"检测到 {len(faces)} 个人脸，请确保画面中只有一个人脸")

        except Exception as e:
            logger.error(f"人脸采集过程出错: {str(e)}")
            return False, None, f"人脸采集错误: {str(e)}"
        finally:
            logger.info("正在释放摄像头资源...")
            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # 确保所有窗口都被正确关闭

    def register_guardian(self, guardian_id: str, name: str, phone: str) -> bool:
        """
        注册监护人信息，包括采集人脸
        """
        # 检查监护人是否已存在
        existing_guardian = self.db.get_guardian_by_id(guardian_id)
        if existing_guardian:
            logger.error(f"监护人ID {guardian_id} 已存在")
            return False

        # 采集人脸
        logger.info("开始人脸采集...")
        success, face_embedding, error_msg = self.capture_face()
        
        if not success:
            logger.error(f"人脸采集失败: {error_msg}")
            return False

        # 将人脸编码转换为字节数据存储
        face_embedding_bytes = face_embedding.tobytes() if face_embedding is not None else None

        # 保存监护人信息到数据库
        if self.db.add_guardian(guardian_id, name, phone, face_embedding_bytes):
            logger.info(f"监护人 {name}（ID：{guardian_id}）注册成功")
            return True
        else:
            logger.error("保存监护人信息失败")
            return False

    def bind_with_student(self, guardian_id: str, student_id: str, relationship: str) -> bool:
        """
        绑定监护人和学生的关系
        """
        # 验证学生和监护人是否存在
        student = self.db.get_student_by_id(student_id)
        guardian = self.db.get_guardian_by_id(guardian_id)

        if not student:
            logger.error(f"学生ID {student_id} 不存在")
            return False

        if not guardian:
            logger.error(f"监护人ID {guardian_id} 不存在")
            return False

        # 添加关系
        if self.db.add_relationship(student_id, guardian_id, relationship):
            logger.info(f"成功绑定关系：{guardian['name']} 是 {student['name']} 的{relationship}")
            return True
        else:
            logger.error("关系绑定失败")
            return False

    def register_from_image(self, image_path, guardian_id, name, phone):
        """从图片注册监护人人脸信息"""
        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")
            
            # 使用 InsightFace 检测和提取特征
            faces = self.face_app.get(image)
            if not faces:
                raise ValueError("图片中未检测到人脸")
            if len(faces) > 1:
                raise ValueError("图片中检测到多个人脸，请提供单人照片")
            
            # 获取人脸特征
            face_embedding = faces[0].embedding
            if face_embedding is None:
                raise ValueError("无法提取人脸特征")
            
            # 将numpy数组转换为bytes格式
            face_embedding_bytes = face_embedding.tobytes()
            
            # 保存到数据库
            self.db.add_guardian(
                guardian_id=guardian_id,
                name=name,
                phone=phone,
                face_encoding=face_embedding_bytes
            )
            
            return True, "监护人注册成功"
        
        except Exception as e:
            logger.error(f"图片注册失败: {str(e)}")
            return False, f"注册失败: {str(e)}" 