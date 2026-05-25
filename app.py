import streamlit as st
import sys
import os
from langchain_core.messages import HumanMessage, AIMessage
from mysql_db import log_conversation

# ==================== 第一部分：环境变量全流程异常捕获 ====================
try:
    from dotenv import load_dotenv
except ImportError:
    print(" 依赖缺失：请先安装 python-dotenv，执行命令：pip install python-dotenv")
    sys.exit(1)

try:
    load_result = load_dotenv(encoding="utf-8", override=True)
    if not load_result:
        print("  警告：未在项目根目录找到 .env 配置文件")

    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

    if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY.strip() == "":
        raise ValueError("DASHSCOPE_API_KEY 未配置或为空，请在 .env 文件中填写有效的通义千问API密钥")

    os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY.strip()
    print(" 环境变量加载校验成功")

    # MySQL 初始化移到这里（环境变量已就绪）
    from mysql_db import init_mysql
    init_mysql()

except Exception as e:
    error_msg = f"环境配置失败：{str(e)}"
    print(f" {error_msg}")
    try:
        st.error(error_msg)
    except:
        pass
    sys.exit(1)

# ==================== 第二部分：导入Agent核心 ====================
try:
    from agent import graph
except Exception as e:
    st.error(f"导入Agent失败：{str(e)}，请检查 agent.py 文件是否存在且无语法错误")
    st.stop()

# ==================== 第三部分：页面配置与初始化 ====================
st.set_page_config(
    page_title="AI 技术学习助手",
    layout="wide"
)

# ---------- 强制用户 ID 输入（必须以 user 开头） ----------
if "user_id" not in st.session_state:
    st.session_state.user_id = None

if st.session_state.user_id is None:
    st.title("👋 欢迎使用 AI 技术学习助手")
    st.markdown("### 请输入你的专属用户 ID")
    st.markdown("**要求：必须以 `user` 开头,尽量使用你名字的小写字母**，例如 你叫小明，那么你的用户 ID 就应该是 `userxiaoming`、`user_xiaoming`")
    st.markdown("_同一 ID 可在不同设备上找回你的历史记忆_")

    with st.form("user_login"):
        user_input = st.text_input("用户 ID", placeholder="例如：userxiaoming")
        submitted = st.form_submit_button("开始对话")
        if submitted:
            if not user_input.strip():
                st.warning("⚠️ 用户 ID 不能为空")
            elif not user_input.strip().startswith("user"):
                st.error("❌ 用户 ID 必须以 `user` 开头，请重新输入")
            else:
                st.session_state.user_id = user_input.strip()
                st.rerun()
    st.stop()

USER_ID = st.session_state.user_id

# 侧边栏显示当前用户和退出选项
with st.sidebar:
    st.markdown(f"### 👤 当前用户：**{USER_ID}**")
    if st.button("切换用户 / 退出"):
        st.session_state.user_id = None
        st.rerun()
    st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("AI 技术学习助手")
st.markdown("---")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==================== 第四部分：核心对话逻辑 ====================
if prompt := st.chat_input("输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在思考..."):
            config = {
                "configurable": {
                    "user_id": USER_ID,
                    "thread_id": "thread1"
                }
            }
            response_placeholder = st.empty()
            full_response = ""

            try:
                for chunk in graph.stream(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=config
                ):
                    for node, data in chunk.items():
                        if data and isinstance(data, dict) and "messages" in data:
                            last_msg = data["messages"][-1]
                            if isinstance(last_msg, AIMessage) and last_msg.content:
                                full_response += last_msg.content
                                response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

            except Exception as e:
                error_str = str(e).lower()
                error_tips = {
                    "invalidapikey": "API密钥无效",
                    "apikeyexpired": "API密钥已过期",
                    "insufficientquota": "API配额不足",
                    "throttling": "请求被限流",
                    "modelnotfound": "模型名称配置错误",
                    "connection": "网络连接失败",
                    "timeout": "请求超时"
                }
                final_tip = "对话处理失败，请稍后重试"
                for key, tip in error_tips.items():
                    if key in error_str:
                        final_tip = tip
                        break
                else:
                    final_tip = f"对话处理失败：{str(e)}"
                response_placeholder.error(final_tip)
                full_response = f"[请求失败] {final_tip}"
                print(f"对话异常详情：{str(e)}")

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # 对话日志写入 MySQL
    try:
        log_conversation(
            session_id=config["configurable"]["thread_id"],
            user_id=config["configurable"]["user_id"],
            role="user",
            content=prompt
        )
        log_conversation(
            session_id=config["configurable"]["thread_id"],
            user_id=config["configurable"]["user_id"],
            role="assistant",
            content=full_response
        )
    except Exception as e:
        print(f"对话日志写入失败: {e}")
# 终端输入测试  streamlit run app.py