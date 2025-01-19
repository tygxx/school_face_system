from app.face_detection import FaceDetector
from app.utils.logger import setup_logger
from app.utils.exceptions import FaceSystemError

logger = setup_logger('main')

def main():
    try:
        # 创建人脸检测器实例
        detector = FaceDetector()
        
        # 启动视频检测
        logger.info("Starting face detection system...")
        logger.info("Press 'q' to quit")
        detector.start_video_detection()
        
    except FaceSystemError as e:
        logger.error(f"System error: {str(e)}")
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
    finally:
        logger.info("System shutdown")

if __name__ == "__main__":
    main()
