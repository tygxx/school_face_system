-- 创建数据库
CREATE DATABASE IF NOT EXISTS school_face COMMENT '校园人脸识别系统数据库';
USE school_face;

-- 创建学生表
CREATE TABLE IF NOT EXISTS students (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
    student_id VARCHAR(50) NOT NULL UNIQUE COMMENT '学号（唯一）',
    name VARCHAR(100) NOT NULL COMMENT '学生姓名',
    class_name VARCHAR(50) NOT NULL COMMENT '班级名称',
    face_encoding MEDIUMBLOB COMMENT '人脸特征编码数据',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间'
) COMMENT '学生信息表';

-- 创建监护人表
CREATE TABLE IF NOT EXISTS guardians (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
    guardian_id VARCHAR(50) NOT NULL UNIQUE COMMENT '监护人ID（唯一）',
    name VARCHAR(100) NOT NULL COMMENT '监护人姓名',
    phone VARCHAR(20) NOT NULL COMMENT '联系电话',
    face_encoding MEDIUMBLOB COMMENT '人脸特征编码数据',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间'
) COMMENT '监护人信息表';

-- 创建学生-监护人关系表
CREATE TABLE IF NOT EXISTS student_guardian_relations (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
    student_id VARCHAR(50) NOT NULL COMMENT '学生学号',
    guardian_id VARCHAR(50) NOT NULL COMMENT '监护人ID',
    relationship VARCHAR(50) NOT NULL COMMENT '关系类型（如：父亲、母亲、祖父等）',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    CONSTRAINT fk_student FOREIGN KEY (student_id) 
        REFERENCES students(student_id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_guardian FOREIGN KEY (guardian_id) 
        REFERENCES guardians(guardian_id) ON DELETE CASCADE ON UPDATE CASCADE,
    UNIQUE KEY unique_relation (student_id, guardian_id) COMMENT '确保学生和监护人关系唯一'
) COMMENT '学生-监护人关系表';

-- 创建访问记录表
CREATE TABLE IF NOT EXISTS access_logs (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
    student_id VARCHAR(50) COMMENT '学生学号',
    guardian_id VARCHAR(50) COMMENT '监护人ID',
    access_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '访问时间',
    event_type ENUM('entry', 'exit', 'pickup') NOT NULL COMMENT '事件类型：entry-进入，exit-离开，pickup-接送',
    authorized BOOLEAN NOT NULL COMMENT '是否授权：true-已授权，false-未授权',
    CONSTRAINT fk_access_student FOREIGN KEY (student_id) 
        REFERENCES students(student_id) ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT fk_access_guardian FOREIGN KEY (guardian_id) 
        REFERENCES guardians(guardian_id) ON DELETE SET NULL ON UPDATE CASCADE
) COMMENT '访问记录表';

-- 添加索引
CREATE INDEX idx_student_id ON students(student_id) COMMENT '学生学号索引';
CREATE INDEX idx_guardian_id ON guardians(guardian_id) COMMENT '监护人ID索引';
CREATE INDEX idx_access_time ON access_logs(access_time) COMMENT '访问时间索引'; 