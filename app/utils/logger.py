import logging
import logging.handlers
import os
from datetime import datetime

def setup_logger(name: str = None) -> logging.Logger:
    """
    配置日志记录器
    :param name: 日志记录器名称
    :return: 配置好的日志记录器
    """
    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # 生成日志文件名
    log_file = os.path.join(log_dir, f'{datetime.now().strftime("%Y-%m-%d")}.log')
    
    # 创建日志记录器
    logger = logging.getLogger(name or __name__)
    logger.setLevel(logging.DEBUG)

    # 文件处理器 - 记录所有日志
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    
    # 控制台处理器 - 只记录INFO及以上级别
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 设置日志格式
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# 创建全局异常处理装饰器
def handle_exceptions(logger: logging.Logger):
    """
    异常处理装饰器
    :param logger: 日志记录器
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator 