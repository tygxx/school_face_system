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