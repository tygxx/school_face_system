# 校园人脸识别系统

这是一个基于人脸识别的校园安全管理系统，用于识别和验证学生与接送家长之间的关系，提高校园安全性。

## 功能特点

- 实时人脸检测和识别
- 学生-家长关系验证
- 未授权接送告警
- 支持多人脸同时识别
- 可视化界面显示

## 系统要求

- Python 3.10 或更高版本
- macOS/Linux/Windows
- 摄像头设备

## 安装步骤

1. 安装 Miniconda（推荐）
```bash
# 下载 Miniconda 安装程序
## macOS ARM (M1/M2)
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh

## macOS Intel
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh

## Linux
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

## Windows
# 访问 https://docs.conda.io/en/latest/miniconda.html 下载安装程序
```

2. 创建并激活虚拟环境
```bash
# 创建环境
conda create -n face_system python=3.10

# 激活环境
conda activate face_system
```

3. 安装依赖
```bash
# 安装 dlib（通过 conda-forge）
conda install -c conda-forge dlib

# 安装其他依赖
pip install opencv-python face-recognition SQLAlchemy python-dotenv
```

## 使用说明

1. 启动系统
```bash
# 确保在 face_system 环境中
conda activate face_system

# 运行程序
python main.py
```

2. 首次运行需要授权摄像头访问权限：
   - **macOS**：
     - 打开系统偏好设置
     - 点击"隐私与安全性"
     - 选择"摄像头"
     - 允许 Python/Terminal 访问摄像头
   - **Windows**：
     - 打开设置
     - 点击"隐私"
     - 选择"摄像头"
     - 允许应用访问摄像头

3. 操作说明
   - 程序启动后会打开摄像头窗口
   - 检测到的人脸会用方框标注
   - 已识别的人脸显示绿色方框和姓名
   - 未识别的人脸显示红色方框
   - 按 'q' 键退出程序

## 项目结构

```
school_face_system/
├── app/
│   └── face_detection.py    # 人脸检测核心模块
├── config/                  # 配置文件目录
├── models/                  # 数据模型
├── web/                    # Web界面相关代码
├── logs/                   # 日志文件
├── main.py                 # 主程序入口
└── requirements.txt        # 项目依赖
```

## 注意事项

1. 确保摄像头正常工作且已授权访问
2. 建议在光线充足的环境下使用
3. 首次运行时需要录入学生和家长的人脸信息
4. 请确保数据库配置正确

## 后续开发计划

- [ ] 添加Web管理界面
- [ ] 实现数据库存储
- [ ] 添加用户认证
- [ ] 优化识别准确度
- [ ] 添加历史记录查询
- [ ] 实现实时告警通知

## 技术支持

如有问题或建议，请提交 Issue 或联系开发团队。 