import os
from app.student_registration import StudentRegistration
from app.guardian_registration import GuardianRegistration

def test_registration():
    # 创建测试目录
    os.makedirs('tests/test_images', exist_ok=True)
    
    # 初始化注册器
    student_reg = StudentRegistration()
    guardian_reg = GuardianRegistration()
    
    # 测试学生注册
    # success, message = student_reg.register_from_image(
    #     image_path="/Users/tengyong/Downloads/IMG_2754.JPG",  # 替换为实际的图片路径
    #     student_id="2024001",
    #     name="张三",
    #     class_name="一年级1班"
    # )
    # print(f"学生注册结果: {success}, {message}")
    
    # 测试监护人注册
    success, message = guardian_reg.register_from_image(
        image_path="/Users/tengyong/Downloads/IMG_2754.JPG",  # 替换为实际的图片路径
        guardian_id="G2024001",
        name="李四",
        phone="13800138000"
    )
    print(f"监护人注册结果: {success}, {message}")

if __name__ == "__main__":
    test_registration() 