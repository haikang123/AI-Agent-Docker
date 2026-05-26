"""
MySQL 数据库持久化模块
用于存储文件同步哈希记录、用户长期记忆、对话历史等，替换原生的 JSON 文件存储。
需提前安装依赖：pip install pymysql sqlalchemy
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, Boolean, text
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.exc import SQLAlchemyError

# 配置日志格式和级别
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 配置区 ====================
# 所有 MySQL 配置在 init_mysql() 中动态读取环境变量，确保加载顺序正确
# 请在 .env 文件中配置以下变量：
#   MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE, MYSQL_CHARSET

# ==================== ORM 模型定义（仅用于自动建表）====================
Base = declarative_base()

class FileHashRecord(Base):
    """文件同步哈希记录表"""
    __tablename__ = "file_hash_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_path = Column(String(500), unique=True, nullable=False)
    file_hash = Column(String(64), nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class UserMemory(Base):
    """用户长期记忆表"""
    __tablename__ = "user_memories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=False, index=True)
    memory_id = Column(String(36), unique=True, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    is_deleted = Column(Boolean, default=False)

class ConversationLog(Base):
    """对话历史日志表"""
    __tablename__ = "conversation_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    user_id = Column(String(100), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.now)

# ==================== 数据库连接管理 ====================
engine = None
SessionLocal = None

def init_mysql():
    """初始化 MySQL 连接并创建所有表"""
    global engine, SessionLocal
    if engine is not None:
        return engine

    mysql_user = os.getenv("MYSQL_USER", "root")
    mysql_password = os.getenv("MYSQL_PASSWORD", "your_password")
    mysql_host = os.getenv("MYSQL_HOST", "127.0.0.1")
    mysql_port = os.getenv("MYSQL_PORT", "3306")
    mysql_database = os.getenv("MYSQL_DATABASE", "ai_assistant")
    mysql_charset = os.getenv("MYSQL_CHARSET", "utf8mb4")
    database_url = f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_database}?charset={mysql_charset}"

    try:
        engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            echo=False
        )
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logger.info("✅ MySQL 数据库连接成功，表已就绪")
        return engine
    except Exception as e:
        logger.warning(f"⚠️ MySQL 连接失败，将回退到本地文件存储。错误：{e}")
        engine = None
        SessionLocal = None
        return None

def _get_session():
    """获取一个新的数据库会话，失败返回 None（内部辅助函数）"""
    import mysql_db as db_module
    if db_module.SessionLocal is None:
        db_module.init_mysql()
    if db_module.SessionLocal is None:
        return None
    return db_module.SessionLocal()

# ==================== 核心操作函数（全部使用原生 SQL）====================
def load_file_hash_records() -> Dict[str, str]:
    """从 MySQL 加载文件哈希记录，返回 {file_path: hash} 字典"""
    import mysql_db as db_module
    session = _get_session()
    if session is None:
        return {}
    try:
        result = session.execute(text("SELECT file_path, file_hash FROM file_hash_records"))
        records = {row[0]: row[1] for row in result}
        return records
    except Exception as e:
        logger.error(f"加载文件哈希记录失败: {e}")
        return {}
    finally:
        session.close()

def update_file_hash_record(file_path: str, file_hash: str) -> bool:
    """更新或插入一条文件哈希记录"""
    import mysql_db as db_module
    session = _get_session()
    if session is None:
        return False
    try:
        session.execute(
            text("INSERT INTO file_hash_records (file_path, file_hash, updated_at) "
                 "VALUES (:fp, :fh, NOW()) "
                 "ON DUPLICATE KEY UPDATE file_hash = :fh2, updated_at = NOW()"),
            {"fp": file_path, "fh": file_hash, "fh2": file_hash}
        )
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"更新文件哈希记录失败: {e}")
        return False
    finally:
        session.close()

def remove_file_hash_record(file_path: str) -> bool:
    """删除某条文件哈希记录"""
    import mysql_db as db_module
    session = _get_session()
    if session is None:
        return False
    try:
        session.execute(
            text("DELETE FROM file_hash_records WHERE file_path = :fp"),
            {"fp": file_path}
        )
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"删除文件哈希记录失败: {e}")
        return False
    finally:
        session.close()

def get_user_memories(user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """获取用户的有效记忆（未软删除），返回字典列表"""
    import mysql_db as db_module
    session = _get_session()
    if session is None:
        return []
    try:
        result = session.execute(
            text("SELECT user_id, memory_id, content, created_at "
                 "FROM user_memories "
                 "WHERE user_id = :uid AND is_deleted = 0 "
                 "ORDER BY created_at DESC LIMIT :lim"),
            {"uid": user_id, "lim": limit}
        )
        memories = []
        for row in result:
            memories.append({
                "user_id": row[0],
                "memory_id": row[1],
                "content": row[2],
                "created_at": row[3]
            })
        return memories
    except Exception as e:
        logger.error(f"获取用户记忆失败: {e}")
        return []
    finally:
        session.close()

def soft_delete_memory(memory_id: str) -> bool:
    """软删除一条记忆"""
    import mysql_db as db_module
    session = _get_session()
    if session is None:
        return False
    try:
        session.execute(
            text("UPDATE user_memories SET is_deleted = 1 WHERE memory_id = :mid"),
            {"mid": memory_id}
        )
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"软删除记忆失败: {e}")
        return False
    finally:
        session.close()

def log_conversation(session_id: str, user_id: str, role: str, content: str) -> bool:
    """记录一条对话日志"""
    import mysql_db as db_module
    session = _get_session()
    if session is None:
        return False
    try:
        session.execute(
            text("INSERT INTO conversation_logs (session_id, user_id, role, content, created_at) "
                 "VALUES (:sid, :uid, :role, :content, NOW())"),
            {"sid": session_id, "uid": user_id, "role": role, "content": content}
        )
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"记录对话日志失败: {e}")
        return False
    finally:
        session.close()

def clean_old_conversation_logs(days: int = 90):
    """删除指定天数前的对话日志"""
    import mysql_db as db_module
    session = _get_session()
    if session is None:
        return
    try:
        cutoff = datetime.now() - timedelta(days=days)
        session.execute(
            text("DELETE FROM conversation_logs WHERE created_at < :cut"),
            {"cut": cutoff}
        )
        session.commit()
        logger.info(f"已清理 {days} 天前的对话日志")
    except Exception as e:
        session.rollback()
        logger.error(f"清理对话日志失败: {e}")
    finally:
        session.close()