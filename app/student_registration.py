import cv2
import face_recognition
import numpy as np
from typing import Optional, Tuple, Dict
from .database import DatabaseManager
from .utils.logger import setup_logger
import os
from .utils.face_detector import FaceDetectorUtil

logger = setup_logger('student_registration')


class StudentRegistration:
    def __init__(self):
        self.db = DatabaseManager()
        try:
            self.face_cascade = FaceDetectorUtil.get_face_cascade()
        except Exception as e:
            print(f"初始化人脸检测器时出错：{str(e)}")
            raise

    def capture_face(self) -> Tuple[bool, Optional[np.ndarray], Optional[str]]:
        """
        采集人脸图像
        返回：(是否成功, 人脸编码数据, 错误信息)
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False, None, "无法打开摄像头"

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    return False, None, "无法读取摄像头画面"

                # 转换为灰度图进行人脸检测
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                # 在画面中标记人脸
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # 显示实时画面
                cv2.imshow('人脸采集 - 按空格键拍照，按Q退出', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    return False, None, "用户取消"
                elif key == ord(' '):  # 按空格键拍照
                    # 检测到人脸才能拍照
                    if len(faces) == 1:
                        # 获取人脸编码
                        face_encodings = face_recognition.face_encodings(frame)
                        if face_encodings:
                            return True, face_encodings[0], None
                        else:
                            return False, None, "无法提取人脸特征"
                    else:
                        logger.warning("请确保画面中只有一个人脸")

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def register_student(self, student_id: str, name: str, class_name: str) -> bool:
        """
        注册学生信息，包括采集人脸
        """
        # 检查学生是否已存在
        existing_student = self.db.get_student_by_id(student_id)
        if existing_student:
            logger.error(f"学号 {student_id} 已存在")
            return False

        # 采集人脸
        logger.info("开始人脸采集...")
        success, face_encoding, error_msg = self.capture_face()
        
        if not success:
            logger.error(f"人脸采集失败: {error_msg}")
            return False

        # 将人脸编码转换为字节数据存储
        face_encoding_bytes = face_encoding.tobytes() if face_encoding is not None else None

        # 保存学生信息到数据库
        if self.db.add_student(student_id, name, class_name, face_encoding_bytes):
            logger.info(f"学生 {name}（学号：{student_id}）注册成功")
            return True
        else:
            logger.error("保存学生信息失败")
            return False 

    def register_from_image(self, image_path, student_id, name, class_name):
        """从图片注册学生人脸信息"""
        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")
            
            # 转换颜色空间（OpenCV使用BGR，face_recognition需要RGB）
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 检测人脸
            face_locations = face_recognition.face_locations(rgb_image)
            if not face_locations:
                raise ValueError("图片中未检测到人脸")
            if len(face_locations) > 1:
                raise ValueError("图片中检测到多个人脸，请提供单人照片")
            
            # 提取人脸特征
            face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
            
            # 将numpy数组转换为bytes格式（与原有录入方式一致）
            face_encoding_bytes = face_encoding.tobytes()
            
            # 保存到数据库
            self.db.add_student(
                student_id=student_id,
                name=name,
                class_name=class_name,
                face_encoding=face_encoding_bytes  # 使用bytes格式，与实时录入保持一致
            )
            
            return True, "学生注册成功"
        
        except Exception as e:
            logger.error(f"图片注册失败: {str(e)}")
            return False, f"注册失败: {str(e)}" 