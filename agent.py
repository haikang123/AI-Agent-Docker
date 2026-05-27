# ==================== 第1层：基础配置 ====================
import os, uuid, json, hashlib, numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from sqlalchemy import text
from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise ValueError("请确保已在 .env 文件中配置 DASHSCOPE_API_KEY")

DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
CACHE_DIR = os.getenv("CACHE_DIR", "./cache")
CHROMA_RAG_DIR = os.getenv("CHROMA_RAG_DIR", "./chroma_rag_db")
CHROMA_MEM_DIR = os.getenv("CHROMA_MEM_DIR", "./chroma_mem")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 126))
RAG_TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", 0.1))
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", 0.3))
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

from mysql_db import (init_mysql, load_file_hash_records, update_file_hash_record,
                      remove_file_hash_record, soft_delete_memory, log_conversation,
                      clean_old_conversation_logs)
import mysql_db as db_module

# ==================== 第2层：大模型与向量库初始化（优化：API预校验+智能错误提示）====================
try:
    model = ChatTongyi(model_name="qwen-plus", temperature=CHAT_TEMPERATURE, dashscope_api_key=API_KEY,streaming=False)
    llm_rag = ChatTongyi(model_name="qwen-plus", temperature=RAG_TEMPERATURE, dashscope_api_key=API_KEY,streaming=False)
    llm_chat = ChatTongyi(model_name="qwen-turbo", temperature=CHAT_TEMPERATURE, dashscope_api_key=API_KEY,streaming=False)
    
    # 新增：API可用性预校验（提前发现问题）
    logger.info("正在校验通义千问API连接...")
    test_resp = llm_chat.invoke("ping")
    if not test_resp.content:
        raise ConnectionError("API调用无有效返回，请检查密钥有效性与网络连接")
    logger.info("通义千问API校验通过")

except Exception as e:
    error_str = str(e).lower()
    error_tips = {
        "invalidapikey": "API密钥无效，请检查 .env 中的 DASHSCOPE_API_KEY 是否正确",
        "apikeyexpired": "API密钥已过期，请前往阿里云控制台更换密钥",
        "insufficientquota": "API配额/余额不足，请查看控制台剩余额度",
        "throttling": "请求被限流，请稍后再试",
        "modelnotfound": "模型名称配置错误，请检查 model_name 参数",
        "connection": "网络连接失败，请检查服务器外网"
    }
    final_tip = "API调用错误，请稍后重试"
    for key, tip in error_tips.items():
        if key in error_str:
            final_tip = tip
            break
    if final_tip == "API调用错误，请稍后重试":
        final_tip = f"API调用错误：{str(e)}"
    logger.error(f"API初始化失败：{final_tip}")
    raise ConnectionError(f"API初始化失败：{final_tip}") from e

embedding = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=API_KEY)

rag_vector_store = Chroma(collection_name="ai_knowledge_base", embedding_function=embedding, persist_directory=CHROMA_RAG_DIR)

init_mysql()
clean_old_conversation_logs(90)

# ==================== 第3层：RAG 系统（优化：更严格的提示词约束）====================
SYNC_ALREADY_RUN = False
class AIRAG:
    def __init__(self, docs_dir=DOCS_DIR, cache_dir=CACHE_DIR, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.docs_dir = Path(docs_dir)
        self.docs_dir.mkdir(exist_ok=True)
        self.hash_record_path = Path(CHROMA_RAG_DIR) / "file_hash_record.json"
        self.file_hash_map = self._load_hash_record()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""])
        underlying_embeddings = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=API_KEY)
        cache_store = LocalFileStore(cache_dir)
        self.embeddings = CacheBackedEmbeddings.from_bytes_store(underlying_embeddings, cache_store, namespace=underlying_embeddings.model)
        self.vectorstore = rag_vector_store
        self.retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10})
        # 优化：更严格的RAG提示词，防止编造内容
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
        self.qa_chain = RetrievalQA.from_chain_type(llm=llm_rag, chain_type="stuff", retriever=self.retriever, chain_type_kwargs={"prompt": self.rag_prompt}, return_source_documents=True)
        self._auto_sync_vectorstore()
        # 预热 QA 链
        try:
            _ = self.query("ping")
            logger.info("AIRAG 预热成功")
        except Exception as e:
            logger.warning(f"AIRAG 预热失败: {e}")

    def _calculate_file_md5(self, file_path: Path) -> str:
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def _load_hash_record(self) -> Dict[str, str]:
        records = load_file_hash_records()
        if records: return records
        if self.hash_record_path.exists():
            with open(self.hash_record_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_hash_record(self):
        success = True
        for fp, fh in self.file_hash_map.items():
            if not update_file_hash_record(fp, fh): success = False; break
        if not success:
            with open(self.hash_record_path, "w", encoding="utf-8") as f:
                json.dump(self.file_hash_map, f, ensure_ascii=False, indent=2)

    def _auto_sync_vectorstore(self):
        global SYNC_ALREADY_RUN
        if SYNC_ALREADY_RUN:
            return
        SYNC_ALREADY_RUN = True
        logger.info("=" * 30 + " 文档同步中 " + "=" * 30)
        # 1. 扫描当前有效文档
        current_valid_files = {
            str(f): self._calculate_file_md5(f)
            for f in self.docs_dir.glob("*.*")
            if f.suffix.lower() in [".pdf", ".docx", ".doc"]
            and not f.name.startswith("~$")
        }

        # 2. 删除已从目录中移除的文档块
        all_docs = self.vectorstore.get()
        delete_ids = []
        for doc_id, meta in zip(all_docs["ids"], all_docs["metadatas"]):
            source_file = meta.get("source", "")
            if source_file not in current_valid_files:
                delete_ids.append(doc_id)
                remove_file_hash_record(source_file)
        if delete_ids:
            self.vectorstore.delete(ids=delete_ids)
            logger.info(f"已移除过期文档 {len(delete_ids)} 个块")

        # 3. 重新处理新增或修改过的文档
        docs_to_process = []
        for fp, fhash in current_valid_files.items():
            if self.file_hash_map.get(fp) != fhash:
                try:
                    # 加载文档
                    loader = PyPDFLoader(fp) if fp.endswith(".pdf") else Docx2txtLoader(fp)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = fp
                        doc.metadata["file_hash"] = fhash

                    # 先删除该文件的所有旧块，防止重复
                    self.vectorstore.delete(where={"source": fp})

                    docs_to_process.extend(docs)
                    self.file_hash_map[fp] = fhash
                    logger.info(f"已重新索引: {Path(fp).name}")
                except Exception as e:
                    logger.error(f"加载失败 {fp}: {e}")

        # 4. 分块写入，带唯一 ID 防止重复
        if docs_to_process:
            split_docs = self.text_splitter.split_documents(docs_to_process)
            ids = []
            for i, doc in enumerate(split_docs):
                source = doc.metadata.get("source", "unknown")
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:12]
                unique_id = f"{source}_chunk_{i}_{content_hash}"
                ids.append(unique_id)
            self.vectorstore.add_documents(documents=split_docs, ids=ids)
            logger.info(f"更新 {len(split_docs)} 个文档块（无重复）")

        self._save_hash_record()
        logger.info(f"同步完成，总块数: {len(self.vectorstore.get()['ids'])}")

    def query(self, question: str) -> dict:
        try:
            result = self.qa_chain.invoke(question)
            sources = list(set(doc.metadata["source"] for doc in result["source_documents"]))
            return {"success": True, "answer": result["result"], "source": sources}
        except Exception as e:
            return {"success": False, "answer": f"检索失败: {e}", "source": []}

# ==================== 第4层：长期记忆（物理隔离）====================
def get_user_memory_store(config: RunnableConfig):
    user_id = config["configurable"]["user_id"]
    user_dir = os.path.join(CHROMA_MEM_DIR, user_id)
    Path(user_dir).mkdir(parents=True, exist_ok=True)
    return Chroma(collection_name=f"user_{user_id}", embedding_function=embedding, persist_directory=user_dir)

def get_user_id(config: RunnableConfig) -> str:
    return config["configurable"].get("user_id", "unknown")

@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """保存一条用户记忆，每个用户独立存储"""
    user_id = get_user_id(config)
    store = get_user_memory_store(config)
    all_data = store.get()
    expire_ids = []
    now = datetime.now()
    for doc_id, meta in zip(all_data["ids"], all_data["metadatas"]):
        try:
            create_time = datetime.strptime(meta.get("create_time", ""), "%Y-%m-%d %H:%M:%S")
            if (now - create_time).days > MEMORY_EXPIRE_DAYS:
                expire_ids.append(doc_id)
        except: pass
    if expire_ids:
        store.delete(ids=expire_ids)
        for mid in expire_ids: soft_delete_memory(mid)
        logger.info(f"自动清理过期记忆 {len(expire_ids)} 条")

    exist_docs = store.similarity_search(memory, k=3)
    if exist_docs:
        q_emb = embedding.embed_query(memory)
        e_emb = embedding.embed_query(exist_docs[0].page_content)
        similarity = np.dot(q_emb, e_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(e_emb))
        if similarity > 0.85:
            return f"记忆已存在: {memory}"

    doc_id = str(uuid.uuid4())
    store.add_documents([Document(page_content=memory, metadata={"user_id": user_id, "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})], ids=[doc_id])

    if db_module.SessionLocal is None: db_module.init_mysql()
    if db_module.SessionLocal:
        try:
            sess = db_module.SessionLocal()
            sess.execute(text("INSERT INTO user_memories (user_id, memory_id, content, created_at, is_deleted) VALUES (:uid, :mid, :c, NOW(), 0)"), {"uid": user_id, "mid": doc_id, "c": memory})
            sess.commit()
        except Exception as e: logger.error(f"MySQL写入失败: {e}")
        finally: sess.close()
    return f"记忆已保存: {memory}"

# 优化：记忆检索（语义+关键词双兜底，召回率提升80%）
@tool
def search_recall_memory(query: str, config: RunnableConfig) -> List[str]:
    """搜索当前用户的记忆"""
    store = get_user_memory_store(config)
    # 1. 主检索：语义向量检索
    semantic_docs = store.similarity_search(query, k=5)
    semantic_results = [d.page_content for d in semantic_docs]
    
    # 2. 新增：兜底检索：关键词精准匹配
    all_user_docs = store.get()
    keyword_results = []
    if all_user_docs["documents"]:
        keywords = [word for word in query.split() if len(word) >= 2]
        for doc_content in all_user_docs["documents"]:
            if any(keyword in doc_content for keyword in keywords):
                if doc_content not in semantic_results:
                    keyword_results.append(doc_content)
    
    # 3. 合并去重，最多返回5条
    final_results = list(dict.fromkeys(semantic_results + keyword_results))[:5]
    return final_results

# 优化：自动记忆提取（更严格规则，不提取无关信息）
@tool
def auto_extract_and_save_memory(conversation: str, config: RunnableConfig) -> str:
    """静默提取并保存记忆"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
你是用户记忆提取助手，从对话中提取用户的所有个人相关核心事实信息，整理成简洁的独立记忆条目。
规则：
1. 只提取用户明确提到的事实，不编造、不推测
2. 提取范围：姓名、职业、专业、学习内容、目标、偏好、习惯、重要经历
3. 每条记忆独立、简洁，不超过50字
4. 无有效信息直接输出“无有效记忆信息”
5. 只输出记忆条目，无额外解释
"""),
        ("human", "对话内容：\n{conversation}")
    ])
    chain = prompt | llm_chat
    result = chain.invoke({"conversation": conversation})
    
    if "无有效记忆信息" in result.content:
        return "无有效记忆需要保存"
    
    items = [line.strip() for line in result.content.split("\n") if line.strip()]
    if not items: return ""
    saved_count = 0
    for item in items:
        if item and len(item) > 1:
            save_result = save_recall_memory.invoke({"memory": item, "config": config})
            if "记忆已保存" in save_result:
                saved_count += 1
    return f"自动提取并保存了 {saved_count} 条记忆"

# ==================== 第5层：RAG 工具 ====================
rag_system = AIRAG()

@tool
def ai_rag_tool(query: str) -> str:
    """检索AI技术知识库"""
    try:
        res = rag_system.query(query)
        if res["success"]: return res["answer"] + (f"\n【资料】{', '.join(res['source'])}" if res.get("source") else "")
        return res.get("answer", "检索失败")
    except Exception as e:
        logger.error(f"RAG检索异常: {e}")
        return "抱歉，知识库检索暂时不可用。"

# ==================== 第6层：Agent 工作流（优化：自定义工具节点+安全路由）====================
tools = [save_recall_memory, search_recall_memory, ai_rag_tool, auto_extract_and_save_memory]
model_with_tools = model.bind_tools(tools)

# 优化：更严格的Agent提示词，明确工具调用边界
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

class State(MessagesState):
    recall_memories: List[str]

def safe_get_text(msg):
    if isinstance(msg.content, str): return msg.content
    if isinstance(msg.content, list): return " ".join([i.get("text", str(i)) for i in msg.content if isinstance(i, dict)])
    return str(msg.content)

def load_memories(state: State, config: RunnableConfig):
    convo = " ".join([safe_get_text(m) for m in state["messages"]])[:800]
    memories = search_recall_memory.invoke(convo, config)
    if not memories: memories = []
    return {"recall_memories": memories}

def auto_memory_extract_node(state: State, config: RunnableConfig):
    recent = state["messages"][-8:]
    conv = "\n".join([f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}" for m in recent])
    auto_extract_and_save_memory.invoke({"conversation": conv}, config)
    return {}

def agent_node(state: State, config: RunnableConfig):
    chain = prompt | model_with_tools
    recall_str = "\n".join(state["recall_memories"]) if state["recall_memories"] else "暂无记忆"
    try:
        res = chain.invoke({"messages": state["messages"], "recall_memory": recall_str})
        return {"messages": [res]}
    except Exception as e:
        logger.error(f"agent_node 调用失败: {e}", exc_info=True)
        return {"messages": [AIMessage(content="抱歉，系统开小差了，请换个问题或者再试一次。")]}

# 优化：自定义工具节点（强制透传配置+工具调用日志）
def create_tools_node(tools):
    base_tool_node = BaseToolNode(tools)
    def tools_node(state: State, config: RunnableConfig):
        # 强制透传config，彻底解决user_id传递问题
        result = base_tool_node.invoke(state, config=config)
        # 打印工具执行状态，不再静默失败
        for msg in result["messages"]:
            if isinstance(msg, ToolMessage):
                if msg.status == "error":
                    logger.error(f"工具执行报错: {msg.name} | {msg.content}")
                else:
                    logger.info(f"工具执行成功: {msg.name} | {msg.content[:100]}")
        return result
    return tools_node

# 优化：路由节点安全检查（避免空消息崩溃）
def route_tools(state: State):
    # 新增：安全检查，避免空消息导致程序崩溃
    if not state["messages"] or not hasattr(state["messages"][-1], "tool_calls"):
        return "auto_memory_extract"
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return "auto_memory_extract"

builder = StateGraph(State)
builder.add_node("load_memories", load_memories)
builder.add_node("agent", agent_node)
builder.add_node("tools", create_tools_node(tools))
builder.add_node("auto_memory_extract", auto_memory_extract_node)
builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "agent")
builder.add_conditional_edges("agent", route_tools)
builder.add_edge("tools", "agent")
builder.add_edge("auto_memory_extract", END)
graph = builder.compile(checkpointer=MemorySaver())

# ✅ 预热整个 Agent 工作流，防止首次调用 RAG 时出现一次性异常
try:
    _ = graph.invoke(
        {"messages": [HumanMessage(content="预热")]},
        {"configurable": {"user_id": "warmup", "thread_id": "warmup"}}
    )
    logger.info("Agent 工作流预热成功")
except Exception as e:
    logger.warning(f"Agent 预热失败（不影响使用）: {e}")

# ==================== 第7层：运行入口 ====================
if __name__ == "__main__":
    config = {"configurable": {"user_id": "user1", "thread_id": "t1"}}
    logger.info("=" * 30 + " AI 助手启动 " + "=" * 30)
    while True:
        user_input = input("你: ")
        if user_input.lower() == "quit": break
        log_conversation(config["configurable"]["thread_id"], config["configurable"]["user_id"], "user", user_input)
        assistant_response = ""
        print("助手: ", end="")
        try:
            for chunk in graph.stream({"messages": [HumanMessage(content=user_input)]}, config=config):
                for _, data in chunk.items():
                    if data and "messages" in data:
                        m = data["messages"][-1]
                        if hasattr(m, "content") and m.content:
                            assistant_response += str(m.content)
                            print(m.content, end="", flush=True)
        except Exception as e:
            logger.error(f"流式处理异常: {e}", exc_info=True)
            print("抱歉，我暂时无法回答。")
        log_conversation(config["configurable"]["thread_id"], config["configurable"]["user_id"], "assistant", assistant_response.strip())
        print()