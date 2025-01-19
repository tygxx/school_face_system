class FaceSystemError(Exception):
    """基础异常类"""
    pass

class CameraError(FaceSystemError):
    """摄像头相关错误"""
    pass

class FaceDetectionError(FaceSystemError):
    """人脸检测错误"""
    pass

class DatabaseError(FaceSystemError):
    """数据库相关错误"""
    pass

class AuthorizationError(FaceSystemError):
    """授权相关错误"""
    pass 