
## 2026-05-23 Gateway Phase 1-7 + Dashboard

### Phase 1 — 配置管理
- `main.py`: TOOL_MODEL / SUMMARY_MODEL 配置项，读写 gateway_config 表
- `main.py`: verify_admin() 保护 /api/settings 端点
- `main.py`: 启动时从 DB 恢复所有 gateway_config
- commits: 177b6a5, 1bf2aa6, da9178e

### Phase 2 — 双模型路由
- `main.py`: consolidation 和缓存摘要改用 TOOL_MODEL
- commit: b7d6c32

### Phase 3 — Status + Activity
- `database.py`: gateway_activity 表 + log_activity()
- `main.py`: GET /api/status, GET /api/activity
- `main.py`: 聊天流程埋 log_activity（收到消息 / 搜记忆 / 调模型 / 回复完成）
- commits: a0779c3, f1d3b57, 08a5fba, 7182fba, cd2cfaf

### Phase 4 — Pending Memory
- `database.py`: pending_memories 表
- `main.py`: GET /api/memory/pending, POST /api/memory/confirm, POST /api/memory/discard
- `main.py`: extract_pending_memories() 自动提取函数，聊天完成后 asyncio.create_task 触发
- `main.py`: POST /api/memory/extract 手动日期范围提取端点
- commits: bbf1d30, ae8daee, c859661

### Phase 5 — 摘要系统（部分）
- `main.py`: user_state_summaries 表 + GET /api/summary/latest + POST /api/summary/generate
- `main.py`: 聊天时自动注入最新摘要到 context
- commits: a60f423, e6b9c34, 2d64621

### Phase 6 — MCP 管理
- `main.py`: POST /api/mcp/test, POST /api/mcp/toggle
- commit: 4950936

### Phase 7 — 巡逻猫猫接入
- `main.py`: 聊天前拉巡逻猫猫实时状态注入 context
- commit: b46d32a

### 其他
- `database.py`: 向量模式搜索合并 core_memories
- commit: 2e179e7

## 2026-05-24 Dashboard 前端 + Phase 3 收尾

### Phase 3 收尾 — 调模型日志
- `main.py`: streaming 和非 streaming 路径各加一行 log_activity("model", ...)
- commit: cd2cfaf

### Dashboard — 状态监控面板
- `templates/dashboard.html`: section-status, sidebar 导航项
- `static/js/dashboard.js`: loadStatusPanel(), loadGatewayStatus(), loadActivityLog(), getAdminHeaders()
- `static/css/dashboard.css`: 状态指标卡片 + 活动日志表格样式
- 对接端点: GET /api/status, GET /api/activity
- commit: aaacc08

### Dashboard — 待审记忆面板
- `templates/dashboard.html`: section-pending, sidebar 导航项
- `static/js/dashboard.js`: loadPendingMemories(), confirmMemory(), discardMemory(), manualExtract()
- `static/css/dashboard.css`: 待审列表 + 操作按钮 + 手动提取表单样式
- 对接端点: GET /api/memory/pending, POST /api/memory/confirm, POST /api/memory/discard, POST /api/memory/extract
- commit: 9faf337

### Dashboard — MCP 管理
- `templates/dashboard.html`: settings 页新增 MCP 管理区块
- `static/js/dashboard.js`: 测试连接 + toggle 开关
- `static/css/dashboard.css`: MCP 表格样式
- 对接端点: POST /api/mcp/test, POST /api/mcp/toggle
- commit: eeaf90c

## Dashboard 完整版上线

### 已推送commit
- c652e7a feat: add TOOL_MODEL and SUMMARY_MODEL selectors to settings
- 6b8c741 fix: load MCP servers from backend and add editable URL fields  
- 3f57c95 feat: add conversation embedding backfill endpoint and dashboard panel

### Dashboard 新面板总结
1. **状态监控** (`section-status`) - GET /api/status + GET /api/activity
2. **待审记忆** (`section-pending`) - GET /api/memory/pending, POST confirm/discard/extract
3. **MCP 管理** (settings 页) - POST /api/mcp/test, POST /api/mcp/toggle
4. **对话向量补算** (记忆管理 tab) - POST /api/admin/embed-conversations + GET status
5. **TOOL_MODEL / SUMMARY_MODEL 选择器** (settings 页) - 用于模型路由

### 技术细节
- ChromaDB 已淘汰，全量使用 PostgreSQL pgvector
- bge-m3 本地部署，embedding 向量维度 256
- conversation_embeddings 新表：存对话 embedding 结果，用于语义搜索
- 对话和记忆的 embedding 后台异步执行，不阻塞主流程
- Dashboard 完全接管配置管理：无需直接改.env，所有设置从 /api/settings 读写

### VPS 验证
- 2026-05-24 13:09 systemctl restart 完成
- 端点测试通过：/api/status (online, 324 memories), /api/activity (empty, 正常)
- Gateway 运行端口：3476

