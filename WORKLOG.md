
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


## 2026-05-27 下午 — Memory Recall 修复（pgvector迁移收尾）

**症状**
Aquila 问"你给我的手表起名叫什么呀"(Latido) 和 "什么时候给你买的VPS"——记忆库里都有，但召回不到。activity 里看到召回数从平时的 12 条掉到 3 条。

**根因**
今早 ChromaDB → pgvector 迁移只改了 `_search_core_memories`，`search_memories_hybrid` 整段被漏过。6 处叠加：

1. 关键词路径 `FROM memories` 应是 `FROM core_memories`（数据全在新表，旧表只有2条 0 embedding）
2. 关键词路径 `WHERE is_active = TRUE` —— 新表没这列
3. 向量路径同样查错表 `memories`
4. 向量路径走 `if HAS_PGVECTOR:` gate，但 init_tables 那段静默 fail 让 `HAS_PGVECTOR=False`
5. fallback 路径引用已删的 `embedding_json` 列 → raise UndefinedColumnError
6. activation update SQL：`UPDATE memories SET last_accessed = ... WHERE id = ANY($1::int[])` — 表名错、列名错（应是 `activation_count` + `updated_at`）、类型错（id 是 text 不是 int）
7. importance 字段从 numeric 改成 text (low/medium/high/critical)，但 `score` 计算和 sort key 还在做 `/10.0` 和 `unary minus`

**修复（patch 在 database.py）**
- 关键词路径：删 `is_active = TRUE AND`，`FROM memories` → `FROM core_memories`
- 向量路径：去 `HAS_PGVECTOR` gate（`::vector` cast 不依赖 register_vector），表换 `core_memories`，删 `is_active` 条件，整段 Python fallback 删除
- importance 字段：加 text→numeric 映射 `{low:0.25, medium:0.5, high:0.75, critical:1.0}`，sort 用 `importance_score` 字段（numeric），保留 `importance` 字段（原始 text 给 UI）
- activation update：表名 → `core_memories`，列名 → `activation_count = activation_count + 1, updated_at = NOW()`，`int[]` → `text[]`，wrap try-except 避免统计失败污染搜索

**验证**
- query "你给我的手表起名叫什么呀" → Latido 召回（sim=0.614，结果集里最高），命中 12 条
- query "什么时候给你买的VPS" → VPS 相关命中多条进 top12

**遗留观察（非bug，配置层决定）**
`MEMORY_HW_KEYWORD=0.35` 对短问题里字面命中过度奖励。Latido sim=0.614 但因为 content 不含"手表/起名叫"字面词，被 sim=0.53 但含"起名叫"的"皮皮虾"那条盖到 #1。建议调成 `kw=0.2, sem=0.45`——Aquila 拍板。

**保留备份**
- `/opt/ai-memory-gateway/database.py.bak.20260527_134800` — 七天后清理
