import cv2
import face_recognition
import numpy as np
from typing import Optional, Tuple
from .database import DatabaseManager
from .utils.logger import setup_logger
from .utils.face_detector import FaceDetectorUtil

logger = setup_logger('guardian_registration')


class GuardianRegistration:
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
                cv2.imshow('监护人人脸采集 - 按空格键拍照，按Q退出', frame)

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
                    elif len(faces) == 0:
                        logger.warning("未检测到人脸，请正对摄像头")
                    else:
                        logger.warning(f"检测到 {len(faces)} 个人脸，请确保画面中只有一个人脸")

        finally:
            cap.release()
            cv2.destroyAllWindows()

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
        success, face_encoding, error_msg = self.capture_face()
        
        if not success:
            logger.error(f"人脸采集失败: {error_msg}")
            return False

        # 将人脸编码转换为字节数据存储
        face_encoding_bytes = face_encoding.tobytes() if face_encoding is not None else None

        # 保存监护人信息到数据库
        if self.db.add_guardian(guardian_id, name, phone, face_encoding_bytes):
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
            self.db.add_guardian(
                guardian_id=guardian_id,
                name=name,
                phone=phone,
                face_encoding=face_encoding_bytes  # 使用bytes格式，与实时录入保持一致
            )
            
            return True, "监护人注册成功"
        
        except Exception as e:
            logger.error(f"图片注册失败: {str(e)}")
            return False, f"注册失败: {str(e)}" 