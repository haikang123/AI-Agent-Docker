# 智学 - AI 技术学习助手

基于 **LangGraph + 通义千问** 构建的智能对话 Agent。支持 **多用户长期记忆（物理隔离）**、**RAG 知识库检索**、**MySQL 双重备份**，以及 **Docker 一键部署**。搭配 **Streamlit** 可视化前端界面，已上线云服务器。

> 🌐 **在线演示**：http://124.220.15.132:8501

---

## 🎯 核心功能

- **🧠 长期记忆（物理隔离）**  
  每个用户独立的 Chroma 向量数据库，从物理层面保证记忆隔离，100% 可靠，不依赖任何第三方库的过滤机制。

- **📚 RAG 知识库问答**  
  基于 Chroma 向量库，支持 PDF/DOCX 文档的知识库检索。MD5 哈希检测文档变更，增量同步。

- **💬 流式输出**  
  Token 级流式输出，打字机效果实时展示 AI 回复。

- **🗄️ MySQL 双重备份**  
  记忆和对话日志同步写入 MySQL，提供结构化查询和持久化保障。过期记忆自动软删除，对话日志定期清理。

- **🛡️ API 异常自动恢复**  
  内置重试与兜底机制，API 偶发故障时自动返回友好提示，不会崩溃。

- **🐳 Docker 一键部署**  
  一条命令即可部署 MySQL + Streamlit 应用，无需手动配置环境。

- **🎨 Streamlit 可视化界面**  
  支持多用户登录切换，零门槛使用。

---

## 🏗️ 系统架构

用户浏览器
↓ HTTP (8501)
Streamlit 前端 (app.py)
↓
LangGraph Agent (agent.py)
├── search_recall_memory → Chroma (每个用户独立DB)
├── ai_rag_tool → Chroma (共享知识库)
├── save_recall_memory → Chroma + MySQL
└── auto_extract_and_save_memory → 大模型提炼 → 保存
↓
MySQL (对话日志、记忆备份、文件哈希)
##  环境准备
- Python 3.10+
- 阿里云通义千问 API 密钥（获取地址：https://dashscope.console.aliyun.com/）


---

## 🚀 快速启动

### 方式一：本地运行

**1. 克隆项目**
```bash
git clone https://github.com/haikang123/AI-agent-demo.git
cd AI-agent-demo
```

### 2. 安装依赖
```
pip install -r requirements.txt
```


### 3. 配置说明
在项目根目录新建 .env 文件，填写：
```
DASHSCOPE_API_KEY="你的API密钥"
USER_ID=user1  # 用户专属记忆ID，首次使用请修改为自己的唯一标识,例如 user_xiaoming
```

### 4. 项目启动
```
streamlit run app.py
```

### 5. 项目结构
```
AI-agent-demo/
├── agent.py                # Agent 核心逻辑（物理隔离记忆、RAG、工具定义）
├── app.py                  # Streamlit 前端界面
├── mysql_db.py             # MySQL 数据库持久化模块
├── Dockerfile              # Docker 镜像构建文件
├── docker-compose.yml      # Docker 服务编排文件
├── requirements.txt        # 项目依赖列表
├── .gitignore              # Git 忽略配置
├── .env.example            # 环境变量配置模板
├── docs/                   # RAG 知识库文档目录
├── chroma_rag_db/          # RAG 向量库（自动生成）
├── chroma_memory_db/       # 长期记忆向量库（每个用户独立子目录）
└── cache/                  # 向量嵌入缓存
```
