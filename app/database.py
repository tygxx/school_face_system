import mysql.connector
from mysql.connector import Error
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from .utils.logger import setup_logger

logger = setup_logger('database')

class DatabaseManager:
    def __init__(self):
        self.config = {
            'host': 'localhost',
            'database': 'school_face',
            'user': 'root',
            'password': 'root'
        }
        self.connection = None
        self.connect()

    def connect(self):
        """建立数据库连接"""
        try:
            self.connection = mysql.connector.connect(**self.config)
            logger.info("数据库连接成功")
        except Error as e:
            logger.error(f"数据库连接失败: {e}")
            raise

    def ensure_connection(self):
        """确保数据库连接有效"""
        try:
            if self.connection and not self.connection.is_connected():
                self.connection.reconnect()
        except Error as e:
            logger.error(f"数据库重连失败: {e}")
            self.connect()

    def add_student(self, student_id: str, name: str, class_name: str, face_encoding: Optional[bytes] = None) -> bool:
        """
        添加学生信息
        """
        self.ensure_connection()
        try:
            cursor = self.connection.cursor()
            sql = """
                INSERT INTO students (student_id, name, class_name, face_encoding)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(sql, (student_id, name, class_name, face_encoding))
            self.connection.commit()
            logger.info(f"成功添加学生: {name}")
            return True
        except Error as e:
            self.connection.rollback()
            logger.error(f"添加学生失败: {e}")
            return False
        finally:
            cursor.close()

    def add_guardian(self, guardian_id: str, name: str, phone: str, face_encoding: Optional[bytes] = None) -> bool:
        """
        添加监护人信息
        """
        self.ensure_connection()
        try:
            cursor = self.connection.cursor()
            sql = """
                INSERT INTO guardians (guardian_id, name, phone, face_encoding)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(sql, (guardian_id, name, phone, face_encoding))
            self.connection.commit()
            logger.info(f"成功添加监护人: {name}")
            return True
        except Error as e:
            self.connection.rollback()
            logger.error(f"添加监护人失败: {e}")
            return False
        finally:
            cursor.close()

    def add_relationship(self, student_id: str, guardian_id: str, relationship: str) -> bool:
        """
        添加学生-监护人关系
        """
        self.ensure_connection()
        try:
            cursor = self.connection.cursor()
            sql = """
                INSERT INTO student_guardian_relations (student_id, guardian_id, relationship)
                VALUES (%s, %s, %s)
            """
            cursor.execute(sql, (student_id, guardian_id, relationship))
            self.connection.commit()
            logger.info(f"成功添加关系: 学生{student_id} - 监护人{guardian_id}")
            return True
        except Error as e:
            self.connection.rollback()
            logger.error(f"添加关系失败: {e}")
            return False
        finally:
            cursor.close()

    def get_student_by_id(self, student_id: str) -> Optional[Dict]:
        """
        根据学号获取学生信息
        """
        self.ensure_connection()
        try:
            cursor = self.connection.cursor(dictionary=True)
            sql = "SELECT * FROM students WHERE student_id = %s"
            cursor.execute(sql, (student_id,))
            result = cursor.fetchone()
            return result
        except Error as e:
            logger.error(f"查询学生信息失败: {e}")
            return None
        finally:
            cursor.close()

    def get_guardian_by_id(self, guardian_id: str) -> Optional[Dict]:
        """
        根据监护人ID获取监护人信息
        """
        self.ensure_connection()
        try:
            cursor = self.connection.cursor(dictionary=True)
            sql = "SELECT * FROM guardians WHERE guardian_id = %s"
            cursor.execute(sql, (guardian_id,))
            result = cursor.fetchone()
            return result
        except Error as e:
            logger.error(f"查询监护人信息失败: {e}")
            return None
        finally:
            cursor.close()

    def verify_relationship(self, student_id: str, guardian_id: str) -> bool:
        """
        验证学生和监护人的关系是否存在
        """
        self.ensure_connection()
        try:
            cursor = self.connection.cursor()
            sql = """
                SELECT COUNT(*) FROM student_guardian_relations 
                WHERE student_id = %s AND guardian_id = %s
            """
            cursor.execute(sql, (student_id, guardian_id))
            count = cursor.fetchone()[0]
            return count > 0
        except Error as e:
            logger.error(f"验证关系失败: {e}")
            return False
        finally:
            cursor.close()

    def get_all_face_encodings(self) -> Dict[str, Dict]:
        """
        获取所有已注册的人脸特征数据
        返回格式: {
            'students': {student_id: face_encoding},
            'guardians': {guardian_id: face_encoding}
        }
        """
        self.ensure_connection()
        result = {'students': {}, 'guardians': {}}
        try:
            cursor = self.connection.cursor()
            
            # 获取学生人脸数据
            cursor.execute("SELECT student_id, face_encoding FROM students WHERE face_encoding IS NOT NULL")
            for student_id, face_encoding in cursor.fetchall():
                result['students'][student_id] = face_encoding

            # 获取监护人人脸数据
            cursor.execute("SELECT guardian_id, face_encoding FROM guardians WHERE face_encoding IS NOT NULL")
            for guardian_id, face_encoding in cursor.fetchall():
                result['guardians'][guardian_id] = face_encoding

            return result
        except Error as e:
            logger.error(f"获取人脸数据失败: {e}")
            return result
        finally:
            cursor.close()

    def __del__(self):
        """
        析构函数，确保关闭数据库连接
        """
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("数据库连接已关闭")
