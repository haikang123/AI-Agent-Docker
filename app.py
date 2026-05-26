import streamlit as st
import sys
import os
import logging
from langchain_core.messages import HumanMessage, AIMessage
from mysql_db import log_conversation

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 第一部分：环境变量全流程异常捕获 ====================
try:
    from dotenv import load_dotenv
except ImportError:
    logger.critical("依赖缺失：请先安装 python-dotenv，执行命令：pip install python-dotenv")
    print(" 依赖缺失：请先安装 python-dotenv，执行命令：pip install python-dotenv")
    sys.exit(1)

try:
    load_result = load_dotenv(encoding="utf-8", override=True)
    if not load_result:
        logger.warning("未在项目根目录找到 .env 配置文件")

    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

    if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY.strip() == "":
        raise ValueError("DASHSCOPE_API_KEY 未配置或为空，请在 .env 文件中填写有效的通义千问API密钥")

    os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY.strip()
    logger.info("环境变量加载校验成功")

    from mysql_db import init_mysql
    init_mysql()

except Exception as e:
    error_msg = f"环境配置失败：{str(e)}"
    logger.error(error_msg)
    try:
        st.error(error_msg)
    except:
        pass
    sys.exit(1)

# ==================== 第二部分：导入Agent核心 ====================
try:
    from agent import graph
except Exception as e:
    logger.error(f"导入Agent失败：{str(e)}")
    st.error(f"导入Agent失败：{str(e)}，请检查 agent.py 文件是否存在且无语法错误")
    st.stop()

# ==================== 第三部分：页面配置与初始化 ====================
st.set_page_config(
    page_title="智学-AI 技术学习助手",
    layout="wide"
)

# ---------- 强制用户 ID 输入 ----------
if "user_id" not in st.session_state:
    st.session_state.user_id = None

if st.session_state.user_id is None:
    st.title("👋 欢迎使用 智学-AI 技术学习助手")
    st.markdown("### 请输入你的专属用户 ID")
    st.markdown("**要求：必须以 `user` 开头**，例如 `userzhangsan`、`user_lisi`")   
    st.markdown("**注意：'user'后不能加空格或者中文**")
    st.markdown("_同一 ID 可在不同设备上找回你的历史记忆_")

    with st.form("user_login"):
        user_input = st.text_input("用户 ID", placeholder="例如：userxiaoming")
        submitted = st.form_submit_button("开始对话")
        if submitted:
            # 去除首尾空格
            uid = user_input.strip()
            # 检查是否为空
            if not uid:
                st.warning("⚠️ 用户 ID 不能为空")
            # 检查是否以 user 开头
            elif not uid.startswith("user"):
                st.error("❌ 用户 ID 必须以 `user` 开头")
            # 检查是否包含空格
            elif " " in uid:
                st.error("❌ 用户 ID 不能包含空格")
            # 检查是否包含非 ASCII 字符（中文、emoji 等）
            elif any(ord(c) > 127 for c in uid):
                st.error("❌ 用户 ID 不能包含中文或特殊符号，请使用英文、数字、下划线")
            else:
                st.session_state.user_id = uid
                st.rerun()
        st.stop()

USER_ID = st.session_state.user_id

# 侧边栏
with st.sidebar:
    st.markdown(f"### 👤 当前用户：**{USER_ID}**")
    if st.button("切换用户"):
        st.session_state.user_id = None
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")

# 消息历史
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("智学-AI 技术学习助手")
st.markdown("---")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==================== 第四部分：对话逻辑（Token级流式，使用 stream_mode="messages"） ====================
if prompt := st.chat_input("输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在思考..."):
            config = {
                "configurable": {
                    "user_id": USER_ID,
                    "thread_id": f"thread_{USER_ID}"
                }
            }

            response_placeholder = st.empty()
            full_response = ""

            try:
                for chunk, metadata in graph.stream(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=config,
                    stream_mode="messages"
                ):
                    if chunk is None:
                        continue
                    if not hasattr(chunk, 'content') or chunk.content is None:
                        continue
                    # 确保 content 是字符串
                    text = ""
                    if isinstance(chunk.content, str):
                        text = chunk.content.strip()
                    elif isinstance(chunk.content, list):
                        text = "".join([str(c) for c in chunk.content if c]).strip()
                    if text:
                        full_response += text
                        response_placeholder.markdown(full_response + "▌")
                # 最终显示完整回复，如果回复为空则显示默认提示
                if full_response.strip():
                    response_placeholder.markdown(full_response)
                else:
                    response_placeholder.markdown("_（助手未生成有效回复）_")
                    
            except Exception as e:
                err = str(e).lower()
                tip = "对话失败"
                if "invalidapikey" in err: tip = "API密钥无效"
                elif "quota" in err: tip = "API配额不足"
                elif "timeout" in err: tip = "请求超时"
                else: tip = f"错误：{str(e)[:50]}"
                logger.error(f"对话异常详情：{str(e)}")
                response_placeholder.error(tip)
                full_response = f"[{tip}]"

    # 只保存有实际内容的助手回复
    if full_response.strip():
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # 保存日志
        try:
            log_conversation(
                session_id=config["configurable"]["thread_id"],
                user_id=USER_ID,
                role="user",
                content=prompt
            )
            log_conversation(
                session_id=config["configurable"]["thread_id"],
                user_id=USER_ID,
                role="assistant",
                content=full_response
            )
        except Exception as e:
            logger.error(f"日志写入失败：{e}")
    else:
        # 如果回复为空，移除刚添加的用户消息
        st.session_state.messages.pop()