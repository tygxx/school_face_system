from app.face_detection import FaceDetector
from app.student_registration import StudentRegistration
from app.utils.logger import setup_logger
from app.utils.exceptions import FaceSystemError

logger = setup_logger('main')


def start_face_detection():
    """启动人脸检测系统"""
    try:
        detector = FaceDetector()
        logger.info("启动人脸检测系统...")
        logger.info("按'q'键退出")
        detector.start_video_detection()
    except FaceSystemError as e:
        logger.error(f"系统错误: {str(e)}")
    except KeyboardInterrupt:
        logger.info("用户终止程序")
    except Exception as e:
        logger.error(f"未预期的错误: {str(e)}", exc_info=True)
    finally:
        logger.info("系统关闭")


def main():
    while True:
        print("\n=== 校园人脸识别系统 ===")
        print("1. 注册新学生")
        print("2. 启动人脸检测")
        print("0. 退出")
        
        choice = input("\n请选择操作: ")
        
        if choice == "1":
            # 学生注册流程
            student_reg = StudentRegistration()
            student_id = input("请输入学号: ")
            name = input("请输入姓名: ")
            class_name = input("请输入班级: ")
            
            print("\n准备开始人脸采集...")
            print("请面对摄像头，确保光线充足，画面中只有一个人脸")
            print("按空格键拍照，按Q键退出")
            
            if student_reg.register_student(student_id, name, class_name):
                print("\n✅ 注册成功！")
            else:
                print("\n❌ 注册失败，请重试")
                
        elif choice == "2":
            # 启动人脸检测系统
            start_face_detection()
            
        elif choice == "0":
            print("\n感谢使用！再见！")
            break
        else:
            print("\n无效的选择，请重试")


if __name__ == "__main__":
    main()
