import cv2
import face_recognition
import numpy as np
from typing import List, Tuple, Dict

from .utils.logger import setup_logger, handle_exceptions
from .utils.exceptions import CameraError, FaceDetectionError

# 创建模块日志记录器
logger = setup_logger('face_detection')

class FaceDetector:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_metadata = []  # 存储人脸对应的身份信息（学生/家长）
        logger.info("FaceDetector initialized")
        
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
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_results = []
            
            for face_encoding, face_location in zip(face_encodings, face_locations):
                # 在已知人脸中查找匹配
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                
                best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None
                
                if best_match_index is not None and matches[best_match_index]:
                    metadata = self.known_face_metadata[best_match_index]
                    logger.debug(f"Face matched with metadata: {metadata}")
                else:
                    metadata = {"status": "unknown"}
                    logger.debug("Unknown face detected")
                
                # 转换回原始图像大小
                top, right, bottom, left = face_location
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # 在图像上绘制方框
                color = (0, 0, 255) if metadata.get("status") == "unknown" else (0, 255, 0)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # 添加标签
                label = metadata.get("name", "Unknown")
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                face_results.append({
                    "location": (top, right, bottom, left),
                    "metadata": metadata
                })
            
            return frame, face_results
            
        except Exception as e:
            raise FaceDetectionError(f"Error processing frame: {str(e)}")

    @handle_exceptions(logger)
    def verify_relationship(self, student_id: str, guardian_id: str) -> bool:
        """
        验证学生和监护人的关系
        这里需要连接到数据库进行验证
        """
        # TODO: 实现数据库查询逻辑
        logger.warning("Relationship verification not implemented yet")
        return False

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
                
                # 显示处理后的画面
                cv2.imshow('Face Detection', processed_frame)
                
                # 检查是否有未授权的接送情况
                for result in face_results:
                    if "student_id" in result["metadata"] and "guardian_id" in result["metadata"]:
                        if not self.verify_relationship(
                            result["metadata"]["student_id"],
                            result["metadata"]["guardian_id"]
                        ):
                            logger.warning("Unauthorized pickup detected!")
                
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