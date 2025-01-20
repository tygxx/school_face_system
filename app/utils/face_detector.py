import cv2
import os

class FaceDetectorUtil:
    _instance = None
    _face_cascade = None
    
    @classmethod
    def get_face_cascade(cls):
        """单例模式获取人脸检测器"""
        if cls._face_cascade is None:
            try:
                # 尝试多种方式找到级联分类器文件
                possible_paths = [
                    # 方法1：在OpenCV包目录下查找
                    os.path.join(os.path.dirname(cv2.__file__), 'data', 
                                'haarcascade_frontalface_default.xml'),
                    # 方法2：常见系统路径
                    '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                    '/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                    # 方法3：当前项目目录
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 
                                'haarcascade_frontalface_default.xml')
                ]
                
                for path in possible_paths:
                    print(f"尝试加载级联分类器：{path}")
                    if os.path.exists(path):
                        cls._face_cascade = cv2.CascadeClassifier(path)
                        if not cls._face_cascade.empty():
                            print(f"成功加载级联分类器：{path}")
                            return cls._face_cascade
                            
                raise Exception("未找到有效的级联分类器文件")
                
            except Exception as e:
                print(f"初始化人脸检测器时出错：{str(e)}")
                raise
                
        return cls._face_cascade 