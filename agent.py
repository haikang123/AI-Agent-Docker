# ==================== 第1层：基础配置  ====================
import os               # 读取系统环境变量、创建文件夹、路径管理
import uuid             # 生成唯一ID（给记忆/文档块编唯一标识，避免重复）
import json             # 保存/读取文件哈希记录（记录哪些文档已同步）
import hashlib          # 计算文件MD5哈希 → 实现【增量同步、不重复入库】
import numpy as np      # 向量数据运算（支撑Chroma向量库存储）
from datetime import datetime  # 时间管理 → 【记忆过期自动清理】
from pathlib import Path       # 优雅处理文件路径（跨Windows/Mac）
from typing import List, Dict       # 类型提示
from sqlalchemy import text   # 用于原生 SQL 插入
# 加载 .env 配置
from dotenv import load_dotenv
load_dotenv(override=True)

# 核心配置读取
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise ValueError(" 请确保已在 .env 文件中配置 DASHSCOPE_API_KEY")

# 目录配置
DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
CACHE_DIR = os.getenv("CACHE_DIR", "./cache")
CHROMA_RAG_DIR = os.getenv("CHROMA_RAG_DIR", "./chroma_rag_db")
CHROMA_MEM_DIR = os.getenv("CHROMA_MEM_DIR", "./chroma_mem")

# 文档分块配置
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 126))

# 模型温度配置
RAG_TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", 0.1))
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", 0.3))

# 记忆管理配置
MEMORY_MAX_COUNT = int(os.getenv("MEMORY_MAX_COUNT", 20))
MEMORY_EXPIRE_DAYS = int(os.getenv("MEMORY_EXPIRE_DAYS", 30))

# ==================== 核心依赖导入 ====================
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA  
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode as BaseToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

# ===== MySQL 集成修改：导入数据库模块 =====
from mysql_db import (
    init_mysql,
    load_file_hash_records,
    update_file_hash_record,
    remove_file_hash_record,
    soft_delete_memory,
    log_conversation,
    clean_old_conversation_logs
)
# 额外导入，用于后续直接获取 SessionLocal
import mysql_db as db_module

# ==================== 第2层：大模型, 统一向量库初始化 ====================
# 大模型实例化
try:
    # 1. 初始化所有通义千问大模型实例
    model = ChatTongyi(model_name="qwen-plus", temperature=CHAT_TEMPERATURE, dashscope_api_key=API_KEY)
    llm_rag = ChatTongyi(model_name="qwen-plus", temperature=RAG_TEMPERATURE, dashscope_api_key=API_KEY)
    llm_chat = ChatTongyi(model_name="qwen-turbo", temperature=CHAT_TEMPERATURE, dashscope_api_key=API_KEY)

    # 2. API可用性预校验（用最快的模型做一次轻量调用，提前发现问题）
    print(" 正在校验通义千问API连接...")
    test_resp = llm_chat.invoke("ping")
    if not test_resp.content:
        raise ConnectionError("API调用无有效返回，请检查密钥有效性与网络连接")
    print(" 通义千问API初始化成功，连接正常")

except Exception as e:
    error_str = str(e).lower()
    error_tips = {
        "invalidapikey": "API密钥无效，请检查 .env 中的 DASHSCOPE_API_KEY 是否正确",
        "apikeyexpired": "API密钥已过期，请前往阿里云控制台更换密钥",
        "insufficientquota": "API配额/余额不足，请查看控制台剩余额度",
        "throttling": "请求被限流，请稍后再试",
        "modelnotfound": "模型名称配置错误，请检查 model_name 参数"
    }
    final_tip = "API调用错误，请稍后重试"
    for key, tip in error_tips.items():
        if key in error_str:
            final_tip = tip
            break
    if final_tip == "API调用错误，请稍后重试":
        final_tip = f"API调用错误：{str(e)}"
    print(f" API初始化失败：{final_tip}")
    raise Exception(f"API初始化失败：{final_tip}") from e

embedding = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=API_KEY)

#统一 Chroma 向量库管理，解决配置分散问题
def init_chroma_store(collection_name: str, persist_dir: str):
    """统一初始化 Chroma 向量库，集中管理配置"""
    Path(persist_dir).mkdir(exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=persist_dir,
    )

# AI 知识库 RAG 向量库
rag_vector_store = init_chroma_store(
    collection_name="ai_knowledge_base",
    persist_dir=CHROMA_RAG_DIR
)

# 用户长期记忆向量库
recall_vector_store = init_chroma_store(
    collection_name="user_long_term_memory",
    persist_dir=CHROMA_MEM_DIR
)

# ===== MySQL 集成修改：初始化数据库连接 =====
init_mysql()

# 启动时清理一次旧对话日志（保留 90 天）
clean_old_conversation_logs(90)

# ==================== 第3层：AI 知识库 RAG 系统 ====================
class AIRAG:
    def __init__(
        self,
        docs_dir: str = DOCS_DIR,
        cache_dir: str = CACHE_DIR,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        self.docs_dir = Path(docs_dir)
        self.docs_dir.mkdir(exist_ok=True)

        # ===== MySQL 集成修改：优先从 MySQL 加载哈希记录，失败回退 JSON =====
        self.hash_record_path = Path(CHROMA_RAG_DIR) / "file_hash_record.json"
        self.file_hash_map = self._load_hash_record()

        # 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )

        # 带缓存的嵌入模型
        underlying_embeddings = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=API_KEY)
        cache_store = LocalFileStore(cache_dir)
        self.embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings, cache_store, namespace=underlying_embeddings.model
        )

        # 绑定统一初始化的向量库
        self.vectorstore = rag_vector_store

        # MMR 多样性检索器
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10}
        )

        # RAG 专业提示词
        self.rag_prompt = PromptTemplate(
            template="""
你是严谨的AI技术知识问答助手，只根据提供的资料回答。
规则：
1. 不编造、不扩展资料外内容
2. 技术术语、代码逻辑用自然语言清晰描述
3. 忽略乱码、无效符号
4. 回答简洁、专业、条理清晰

参考资料：
{context}

用户问题：{question}
回答：""",
            input_variables=["context", "question"]
        )

        # RAG 查询链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm_rag,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.rag_prompt},
            return_source_documents=True
        )

        # 启动时自动同步文档
        self._auto_sync_vectorstore()

    def _calculate_file_md5(self, file_path: Path) -> str:
        """计算文件 MD5 哈希，检测内容变更"""
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def _load_hash_record(self) -> Dict[str, str]:
        # ===== MySQL 集成修改：优先从 MySQL 读取 =====
        records = load_file_hash_records()
        if records:
            return records
        # 降级：读取本地 JSON
        if self.hash_record_path.exists():
            with open(self.hash_record_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_hash_record(self):
        # ===== MySQL 集成修改：写入 MySQL，失败则回退 JSON =====
        success = True
        for fp, fh in self.file_hash_map.items():
            if not update_file_hash_record(fp, fh):
                success = False
                break
        if not success:
            # MySQL 写入失败，使用 JSON 文件兜底
            with open(self.hash_record_path, "w", encoding="utf-8") as f:
                json.dump(self.file_hash_map, f, ensure_ascii=False, indent=2)

    def _auto_sync_vectorstore(self):
        """自动同步向量库：增量更新、过期删除、修改重入"""
        print("="*30 + " AI 知识库文档自动同步中 " + "="*30)

        # 1. 扫描当前有效文档
        current_valid_files = {}
        for file in self.docs_dir.glob("*.*"):
            suffix = file.suffix.lower()
            if suffix in [".pdf", ".docx", ".doc"]:
                current_valid_files[str(file)] = self._calculate_file_md5(file)

        # 2. 清理已删除文档的过期数据
        all_stored_docs = self.vectorstore.get()
        delete_ids = []
        for doc_id, metadata in zip(all_stored_docs["ids"], all_stored_docs["metadatas"]):
            source_file = metadata.get("source", "")
            if source_file not in current_valid_files:
                delete_ids.append(doc_id)
                # ===== MySQL 集成修改：同步删除 MySQL 中的哈希记录 =====
                remove_file_hash_record(source_file)

        if delete_ids:
            self.vectorstore.delete(ids=delete_ids)
            print(f" 已移除过期文档，共删除 {len(delete_ids)} 个文档块")

        # 3. 处理新增/修改的文档
        docs_to_process = []
        for file_path, file_hash in current_valid_files.items():
            if self.file_hash_map.get(file_path, "") != file_hash:
                file = Path(file_path)
                try:
                    if file.suffix.lower() == ".pdf":
                        loader = PyPDFLoader(str(file))
                    else:
                        loader = Docx2txtLoader(str(file))

                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = str(file)
                        doc.metadata["file_hash"] = file_hash
                    docs_to_process.extend(docs)

                    self.file_hash_map[file_path] = file_hash
                    print(f" 已加载变更文档: {file.name}")
                except Exception as e:
                    print(f" 文档加载失败 {file.name}: {str(e)}")

        # 4. 分块写入向量库
        if docs_to_process:
            split_docs = self.text_splitter.split_documents(docs_to_process)
            self.vectorstore.add_documents(split_docs)
            print(f" 已写入向量库，共新增 {len(split_docs)} 个文档块")

        # 5. 保存哈希记录
        self._save_hash_record()
        print(f" 同步完成！向量库当前总文档块数: {len(self.vectorstore.get()['ids'])}")

    def query(self, question: str) -> dict:
        """执行 RAG 专业查询"""
        try:
            result = self.qa_chain.invoke(question)
            source_files = list(set([doc.metadata["source"] for doc in result["source_documents"]]))
            return {
                "success": True,
                "answer": result["result"],
                "source": source_files
            }
        except Exception as e:
            return {
                "success": False,
                "answer": f"检索失败: {str(e)}",
                "source": []
            }

# ==================== 第4层：长期记忆系统 ====================
def get_user_id(config: RunnableConfig) -> str:
    """从配置中获取用户 ID，实现用户隔离"""
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("需要用户提供 ID 才能进行记忆操作")
    return user_id

#记忆保存：新增自动过期清理、重复校验，解决冗余问题
@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """保存记忆内容到向量存储"""
    user_id = get_user_id(config)

    # 1. 自动清理过期记忆
    all_user_docs = recall_vector_store.get(where={"user_id": {"$eq": user_id}})
    expire_ids = []
    now = datetime.now()
    for doc_id, metadata in zip(all_user_docs["ids"], all_user_docs["metadatas"]):
        create_time_str = metadata.get("create_time", "")
        try:
            create_time = datetime.strptime(create_time_str, "%Y-%m-%d %H:%M:%S")
            if (now - create_time).days > MEMORY_EXPIRE_DAYS:
                expire_ids.append(doc_id)
        except:
            continue

    if expire_ids:
        recall_vector_store.delete(ids=expire_ids)
        # 同步软删除 MySQL 中的对应记录
        for mem_id in expire_ids:
            soft_delete_memory(mem_id)
        print(f" 自动清理过期记忆 {len(expire_ids)} 条")

    # 2. 重复记忆校验，避免冗余
    exist_docs = recall_vector_store.similarity_search(memory, k=1, filter={"user_id": {"$eq": user_id}})
    if exist_docs:
        query_embedding = embedding.embed_query(memory)
        exist_embedding = embedding.embed_query(exist_docs[0].page_content)
        similarity = np.dot(query_embedding, exist_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(exist_embedding))
        if similarity > 0.85:
            return f" 记忆已存在，无需重复保存: {memory}"

    # 3. 正常保存记忆
    doc = Document(
        page_content=memory,
        id=str(uuid.uuid4()),
        metadata={
            "user_id": user_id,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    )
    recall_vector_store.add_documents([doc])

    # MySQL 结构化备份（原生 SQL 插入）
    if db_module.SessionLocal is None:
        db_module.init_mysql()
    if db_module.SessionLocal is not None:
        session = None
        try:
            session = db_module.SessionLocal()
            session.execute(
                text("INSERT INTO user_memories (user_id, memory_id, content, metadata_json) "
                     "VALUES (:uid, :mid, :c, :m)"),
                {"uid": user_id, "mid": doc.id, "c": memory, "m": json.dumps(doc.metadata)}
            )
            session.commit()
            print(f"✅ 记忆已存入 MySQL: {memory[:50]}{'...' if len(memory) > 50 else ''}")
        except Exception as e:
            print(f"MySQL 记忆写入失败: {e}")
            if session:
                session.rollback()
        finally:
            if session:
                session.close()

    return f" 记忆已保存: {memory}"

# 记忆检索：语义 + 关键词兜底，解决召回率低问题
@tool
def search_recall_memory(query: str, config: RunnableConfig) -> List[str]:
    """搜索相关记忆内容"""
    user_id = get_user_id(config)

    # 1. 主检索：语义向量检索
    semantic_docs = recall_vector_store.similarity_search(query, k=5, filter={"user_id": {"$eq": user_id}})
    semantic_results = [doc.page_content for doc in semantic_docs]

    # 2. 兜底检索：关键词精准匹配
    all_user_docs = recall_vector_store.get(where={"user_id": {"$eq": user_id}})
    keyword_results = []
    if all_user_docs["documents"]:
        keywords = [word for word in query.split() if len(word) >= 2]
        for doc_content in all_user_docs["documents"]:
            if any(keyword in doc_content for keyword in keywords):
                if doc_content not in semantic_results:
                    keyword_results.append(doc_content)

    # 3. 合并去重，最多返回 5 条
    final_results = list(dict.fromkeys(semantic_results + keyword_results))[:5]
    return final_results

#自动记忆提取工具，解决依赖大模型主动触发问题
@tool
def auto_extract_and_save_memory(conversation: str, config: RunnableConfig) -> str:
    """自动从对话中提取用户核心信息并保存到长期记忆"""
    extract_prompt = ChatPromptTemplate.from_messages([
        ("system", """
你是用户记忆提取助手，从对话中提取用户的所有个人相关核心事实信息，整理成简洁的独立记忆条目。
规则：
1. 只提取用户明确提到的事实，不编造、不推测
2. 提取范围：姓名、职业、专业、学习内容、目标、偏好、习惯、重要经历
3. 每条记忆独立、简洁，不超过 50 字
4. 无有效信息直接输出“无有效记忆信息”
5. 只输出记忆条目，无额外解释
"""),
        ("human", "对话内容：\n{conversation}")
    ])
    extract_chain = extract_prompt | llm_chat
    extract_result = extract_chain.invoke({"conversation": conversation})

    if "无有效记忆信息" in extract_result.content:
        return "无有效记忆需要保存"

    # 逐条保存记忆
    memory_items = [line.strip() for line in extract_result.content.split("\n") if line.strip()]
    saved_count = 0
    for item in memory_items:
        save_result = save_recall_memory.invoke({"memory": item, "config": config})
        if "记忆已保存" in save_result:
            saved_count += 1
    return f"自动提取并保存了 {saved_count} 条记忆: {'；'.join(memory_items)}"

# ==================== 第5层：AI 知识库 RAG 工具 ====================
rag_system = AIRAG()

@tool
def ai_rag_tool(query: str) -> str:
    """
    【仅AI技术问题使用】检索深度学习、大模型、RAG、Agent、LangChain、Python开发、AI岗面试等相关技术内容。
    仅用户提问AI相关技术问题时调用，非技术问题禁止调用。
    """
    result = rag_system.query(query)
    if result["success"]:
        source_str = f"\n\n【资料来源】: {', '.join(result['source'])}" if result["source"] else ""
        return result["answer"] + source_str
    return result["answer"]

# ==================== 第6层：Agent 核心流程  ====================
# 工具列表
tools = [save_recall_memory, search_recall_memory, ai_rag_tool, auto_extract_and_save_memory]
model_with_tools = model.bind_tools(tools)

# 自定义工具节点：强制透传 RunnableConfig，捕获工具报错
def create_tools_node(tools):
    base_tool_node = BaseToolNode(tools)

    def tools_node(state: State, config: RunnableConfig):
        # 强制透传 config，彻底解决 user_id 传递问题
        result = base_tool_node.invoke(state, config=config)

        # 打印工具执行状态，不再静默失败
        for msg in result["messages"]:
            if isinstance(msg, ToolMessage):
                if msg.status == "error":
                    print(f" 工具执行报错: {msg.content}")
                else:
                    print(f" 工具执行成功: {msg.name} | {msg.content}")
        return result

    return tools_node

# Agent 提示词
prompt = ChatPromptTemplate.from_messages([
    ("system", """
你是一个具备长期记忆能力的智能助手，同时拥有AI技术知识检索能力，严格遵守以下规则：
1. 【记忆检索强制要求】：回答用户任何问题前，必须先调用 search_recall_memory 工具，检索用户的历史长期记忆
2. 【技术问题规则】：仅当用户提问深度学习、大模型、RAG、Agent、LangChain、Python开发、AI岗面试等相关技术内容时，必须调用 ai_rag_tool 工具，优先使用检索到的技术内容回答
3. 【回答要求】：必须结合检索到的长期记忆回答，不要生硬复读记忆，自然融合到回答中；保持口语化、自然流畅
4. 【工具调用要求】：严格按照工具定义的参数格式调用，不要编造参数

用户历史记忆：{recall_memory}
"""),
    ("placeholder", "{messages}")
])

# Agent 状态定义
class State(MessagesState):
    recall_memories: List[str]

# 辅助函数，用于安全提取消息中的文本内容
def safe_get_text(msg):
    """安全提取消息中的文本内容，兼容多模态/特殊格式"""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # 处理多模态格式，提取所有文本
        texts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
            elif isinstance(item, str):
                texts.append(item)
        return " ".join(texts)
    elif content is None:
        return ""
    else:
        # 其他类型强制转字符串，避免报错
        return str(content)

# 前置节点：强制加载用户记忆
def load_memories(state: State, config: RunnableConfig):
    convo_str = "\n".join([safe_get_text(msg) for msg in state["messages"]])
    convo_str = convo_str[:800]  # 简单截断防止上下文过长
    recall_memories = search_recall_memory.invoke(convo_str, config)
    return {"recall_memories": recall_memories}

# 自动记忆提取节点：对话结束前强制提取记忆，不依赖大模型主动调用
def auto_memory_extract_node(state: State, config: RunnableConfig):
    # 【优化】仅获取最近 4 轮对话（8条消息）进行记忆提取，限制上下文窗口
    recent_messages = state["messages"][-8:] if len(state["messages"]) > 8 else state["messages"]

    conversation = "\n".join([
        f"{'用户' if isinstance(msg, HumanMessage) else '助手'}: {msg.content}"
        for msg in recent_messages
    ])
    auto_extract_and_save_memory.invoke({"conversation": conversation}, config)
    return {}

# Agent 核心决策节点
def agent_node(state: State, config: RunnableConfig):
    bound_prompt = prompt | model_with_tools
    recall_str = "\n".join(state["recall_memories"])
    response = bound_prompt.invoke({
        "messages": state["messages"],
        "recall_memory": recall_str
    })
    return {"messages": [response]}

# 路由节点：判断是否调用工具
def route_tools(state: State):
    # 增加安全检查
    if not state["messages"] or not hasattr(state["messages"][-1], "tool_calls"):
        return "auto_memory_extract"
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return "auto_memory_extract"

# 构建 Agent 流程图
builder = StateGraph(State)

# 节点注册
builder.add_node("load_memories", load_memories)
builder.add_node("agent", agent_node)
builder.add_node("tools", create_tools_node(tools))
builder.add_node("auto_memory_extract", auto_memory_extract_node)

# 流程拓扑
builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "agent")
builder.add_conditional_edges(
    "agent",
    route_tools,
    {"tools": "tools", "auto_memory_extract": "auto_memory_extract"}
)
builder.add_edge("tools", "agent")
builder.add_edge("auto_memory_extract", END)

# 编译 Agent，启用对话状态持久化
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# ==================== 第7层：流式输出 & 运行入口 ====================
# 流式输出：展示工具调用状态，不再屏蔽报错
def stream_output(chunk):
    for node, data in chunk.items():
        if data and isinstance(data, dict) and "messages" in data:
            msg = data["messages"][-1]
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    print(f"\n 正在调用工具: {tool_call['name']} | 参数: {tool_call['args']}")
                continue
            if hasattr(msg, "content") and msg.content:
                print(msg.content, end="", flush=True)

# 本地运行入口
if __name__ == "__main__":
    config = {"configurable": {"user_id": "user1", "thread_id": "thread1"}}

    print("\n" + "="*30 + " AI 技术学习助手 启动完成 " + "="*30)
    print(" 提示：将 AI 技术 PDF/DOCX 文档放入 ./docs 文件夹，程序会自动同步到知识库")
    print(" 输入 quit 退出程序\n")

    while True:
        user_input = input("你: ")
        if user_input.lower() == "quit":
            print(" 对话结束，欢迎下次使用")
            break

        log_conversation(
            session_id=config["configurable"]["thread_id"],
            user_id=config["configurable"]["user_id"],
            role="user",
            content=user_input
        )

        assistant_response = ""
        print("助手: ", end="", flush=True)
        for chunk in graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        ):
            stream_output(chunk)
            for node, data in chunk.items():
                if data and isinstance(data, dict) and "messages" in data:
                    msg = data["messages"][-1]
                    if hasattr(msg, "content") and msg.content:
                        assistant_response += str(msg.content)

        if assistant_response:
            log_conversation(
                session_id=config["configurable"]["thread_id"],
                user_id=config["configurable"]["user_id"],
                role="assistant",
                content=assistant_response.strip()
            )

        print("\n")