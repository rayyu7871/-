# 财富风险诊断访谈系统 — 完整架构设计文档 v1.0

> 最后更新：2026-03-02

---

## 一、系统概述

| 项目 | 说明 |
|------|------|
| 系统名称 | 财富风险诊断访谈系统 |
| 目标 | 通过实时语音对话，在 45-60 分钟内完成客户财富风险全维度诊断 |
| 核心交互 | 语音问答（AI 提问，客户语音回答） |
| 最终产出 | 结构化 PDF 诊断报告 |
| 目标用户 | 财富管理机构的潜在/存量客户 |

---

## 二、技术选型

```
层级          组件                版本/来源           选型理由
──────────────────────────────────────────────────────────────────────
前端          FastAPI             0.115+             轻量、原生异步
              HTML/JS             原生               无框架依赖
              LiveKit Client JS   latest             WebRTC 封装，内置 AEC

音频传输       LiveKit Server      latest             WebRTC 基础设施
                                                    内置 AEC/AGC/NS/抖动缓冲

VAD           Silero VAD          v5                 本地推理，<1ms，最准确

ASR           FunASR SenseVoice   latest             中文最优，含情绪输出
              (Paraformer-zh 流式)                    首选 SenseVoice，降级用 Paraformer

TTS           CosyVoice 2         latest             中文自然度最佳，首包<100ms
                                                     预生成缓存，访谈期间0延迟

LLM（判断）    Qwen-turbo          API                快速(<300ms)，聚焦判断
LLM（提取）    Qwen-plus           API                准确，后台非阻塞
LLM（兜底）    Qwen-max            API                复杂场景兜底

状态机         纯 Python           dataclass+enum     零依赖，轻量透明
诊断引擎       纯 Python 规则       -                  可解释，无需LLM

报告           Jinja2 + WeasyPrint -                  PDF 生成，无服务依赖

存储           PostgreSQL          16                 会话持久化，支持续接
               Redis               7                  音频缓存，会话状态

──────────────────────────────────────────────────────────────────────
移除           LiteLLM / LangGraph / Gradio / Qdrant / REACT Agent
```

---

## 三、整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                          客户端（浏览器）                              │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  LiveKit Client JS                                          │   │
│  │  麦克风采集（WebRTC）    音频播放（WebRTC）    UI 状态展示     │   │
│  └──────────────┬──────────────────────┬────────────────────--─┘   │
└─────────────────│──────────────────────│───────────────────────────┘
                  │ WebRTC（双向音频）    │
                  │ + DataChannel（事件）│
┌─────────────────▼──────────────────────▼───────────────────────────┐
│                       LiveKit Server（WebRTC 网关）                   │
│           AEC / AGC / NS / 抖动缓冲 / DTLS 加密                      │
└─────────────────┬──────────────────────┬───────────────────────────┘
                  │ PCM 音频流           │ 音频帧推送
┌─────────────────▼──────────────────────▼───────────────────────────┐
│                         FastAPI 主服务                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     语音管道层                                │   │
│  │                                                             │   │
│  │  入站：PCM → Silero VAD → FunASR SenseVoice 流式 → 文本队列  │   │
│  │                              ↓ 同时输出                     │   │
│  │                         emotion: sad/anxious/neutral        │   │
│  │                                                             │   │
│  │  出站：音频缓冲区 → LiveKit → 客户端播放                      │   │
│  └───────────────────────────┬─────────────────────────────────┘   │
│                              │ asyncio.Queue（文本 + 情绪）          │
│  ┌───────────────────────────▼─────────────────────────────────┐   │
│  │                     访谈引擎层                                │   │
│  │                                                             │   │
│  │  ┌──────────────────┐   ┌────────────────────────────────┐  │   │
│  │  │ Barge-in 控制器  │   │  Python 状态机                  │  │   │
│  │  │ VAD检测→打断→停播 │   │  InterviewStateMachine          │  │   │
│  │  └──────────────────┘   └──────────────┬─────────────────┘  │   │
│  │                                        │                    │   │
│  │  ┌─────────────────────────────────────▼─────────────────┐  │   │
│  │  │              LLM 判断（Qwen-turbo）                    │  │   │
│  │  │  输入：unit_history + user_text + emotion              │  │   │
│  │  │  输出：{follow_up, next_q_id, unit_complete}           │  │   │
│  │  └─────────────────────────────────────┬─────────────────┘  │   │
│  │                                        │                    │   │
│  │  ┌─────────────────────────────────────▼─────────────────┐  │   │
│  │  │              题库查找（内存，0ms）                       │  │   │
│  │  │  → 取预生成 TTS 音频 → push 到音频缓冲区                 │  │   │
│  │  └────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │ 异步后台                             │
│  ┌───────────────────────────▼─────────────────────────────────┐   │
│  │                   信息提取层（后台并发）                       │   │
│  │  Qwen-plus → 全局字段提取 → 更新 InterviewState              │   │
│  │  触发条件评估 → 动态解锁扩展层单元                             │   │
│  └───────────────────────────┬─────────────────────────────────┘   │
└──────────────────────────────│─────────────────────────────────────┘
                               │ 访谈结束
┌──────────────────────────────▼─────────────────────────────────────┐
│                          诊断 & 报告层                               │
│    规则评分引擎 → 260单元评分 → Jinja2 模板 → WeasyPrint → PDF       │
└─────────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────────┐
│                          持久化层                                    │
│    PostgreSQL（会话/提取结果/报告）    Redis（音频缓存/会话状态）       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 四、核心数据模型

```python
# ── 题库模型 ──────────────────────────────────────────────

class QuestionTier(str, Enum):
    CORE = "core"         # 必问（~30单元）
    EXTENDED = "extended" # 条件触发（~230单元）

class Question(BaseModel):
    question_id: str          # "unit_01_q1"
    text: str
    audio_path: str           # 预生成音频路径
    covers_fields: list[str]  # 此题覆盖哪些字段
    follow_up_ids: list[str]  # 追问题 ID

class RiskUnit(BaseModel):
    unit_id: str
    name: str
    tier: QuestionTier
    trigger_condition: str | None  # "已婚 AND 有子女"（扩展层）
    questions: list[Question]
    required_fields: list[str]
    scoring_rule: dict

# ── 会话状态 ──────────────────────────────────────────────

class InterviewStatus(str, Enum):
    GREETING    = "greeting"
    INTERVIEWING = "interviewing"
    DIAGNOSING  = "diagnosing"
    DONE        = "done"

class InterviewState(BaseModel):
    # 基础信息
    session_id: str
    status: InterviewStatus

    # 访谈进度
    units_queue: list[str]       # 待访问单元（动态，可插队）
    current_unit_id: str
    current_question_id: str
    completed_unit_ids: list[str]

    # 已提取信息（全局，跨单元共享）
    extracted: dict[str, Any]    # field_name → value
    # 示例：{"月收入": 50000, "婚姻状态": "已婚", "子女数量": 2}

    # 对话历史
    unit_history: list[dict]     # 当前单元（LLM判断用，3-5轮）
    full_history: list[dict]     # 全量（报告用）

    # 情绪轨迹
    emotion_log: list[dict]      # [{turn, emotion, text}]

# ── LLM 判断输出 ──────────────────────────────────────────

class TurnDecision(BaseModel):
    action: Literal["follow_up", "next_question", "next_unit", "empathy_first"]
    next_question_id: str
    unit_complete: bool
    reason: str                  # 日志用
```

---

## 五、分层架构详解

### 5.1 前端层

```
职责：音频 I/O + 极简 UI

技术：
  - LiveKit Client JS SDK（WebRTC，含 AEC/AGC）
  - 原生 HTML/JS，FastAPI 托管静态文件
  - DataChannel 接收文本事件（字幕、状态提示）

页面结构：
  ┌─────────────────────────────┐
  │  [波形动画]  系统说话时跳动   │
  │                             │
  │  当前问题字幕（可选）         │
  │                             │
  │  进度：第 12 / 约 30 个话题  │
  │                             │
  │  [麦克风状态指示]            │
  └─────────────────────────────┘

DataChannel 事件（server → client）：
  {type: "subtitle",   text: "..."}      # 当前问题字幕
  {type: "progress",   current: 12, total: 30}
  {type: "empathy",    emotion: "sad"}   # 触发情绪 UI
  {type: "done",       report_url: "..."} # 访谈结束
```

### 5.2 语音管道层

```
入站处理（用户说话）：

  LiveKit → PCM(16kHz, 16bit, mono)
       ↓
  Silero VAD（20ms帧，本地）
       ├── 说话中 → feed FunASR 流式接口
       │           → 实时中间结果（可选显示字幕）
       └── 静音 > 300ms → 触发 ASR finalize
                         → 最终文本 + emotion
                         → push asr_queue

出站处理（系统说话）：

  audio_buffer → PCM bytes
       ↓
  Barge-in 控制器（实时监听 VAD）
       ├── 无打断 → 正常推流到 LiveKit
       └── 检测到用户说话 → 立即停止推流
                          → 清空当前音频队列
                          → 标记「被打断」状态

Barge-in 实现：
  async def stream_audio(audio: bytes):
      chunk_size = 960  # 20ms @ 48kHz
      for i in range(0, len(audio), chunk_size):
          if barge_in_detected.is_set():
              barge_in_detected.clear()
              return  # 立即停止
          await livekit_room.publish_audio(audio[i:i+chunk_size])
          await asyncio.sleep(0.02)
```

### 5.3 访谈引擎层（三个并行协程）

```python
# ── 协程一：判断循环（关键路径）─────────────────────

async def decision_loop():
    while state.status == INTERVIEWING:
        # 等用户说完
        user_text, emotion = await asr_queue.get()

        # 情绪处理（优先）
        if emotion in ("sad", "anxious") and not recently_empathized:
            audio = empathy_audio[emotion]
            await audio_buffer.put(audio)
            recently_empathized = True
            continue  # 本轮不追问，先共情

        # 后台提取（不等结果）
        asyncio.create_task(extract_queue.put({
            "text": user_text,
            "unit_id": state.current_unit_id
        }))

        # LLM 判断（~300ms）
        decision = await llm_judge(state, user_text, emotion)

        # 状态机转移
        state = state_machine.transition(state, decision)

        # 取音频（0ms）→ 放入缓冲
        audio = question_bank.get_audio(decision.next_question_id)
        await audio_buffer.put(audio)

# ── 协程二：语音循环（实时）──────────────────────────

async def voice_loop():
    async for pcm_chunk in livekit.subscribe():
        vad.process(pcm_chunk)
        asr.feed(pcm_chunk)

        if vad.is_speech_end():
            text, emotion = await asr.finalize()
            await asr_queue.put((text, emotion))

        if vad.is_speech_start():
            barge_in_detected.set()  # 触发打断

# ── 协程三：提取循环（后台）──────────────────────────

async def extraction_loop():
    while True:
        task = await extract_queue.get()
        result = await llm_extract(task)   # Qwen-plus

        # 更新全局提取状态
        state.extracted.update(result)

        # 检查是否解锁新的扩展单元
        newly_unlocked = evaluate_triggers(state.extracted, all_units)
        for unit in newly_unlocked:
            if unit.unit_id not in state.units_queue:
                state.units_queue.insert(next_position, unit.unit_id)
```

### 5.4 LLM 层

```
两个 LLM 调用，职责严格分离：

┌────────────────────────────────────────────────────────────┐
│  LLM-1：判断（Qwen-turbo）                                  │
│                                                            │
│  输入：当前单元最近3轮对话 + 用户本轮回答 + 情绪              │
│  输出：{action, next_question_id, unit_complete}            │
│  目标延迟：< 300ms                                          │
│  不做提取，不做评分，只做决策                                │
│                                                            │
│  Prompt 结构：                                             │
│    [系统]：你是访谈决策助手，只判断下一步行动                │
│    [当前单元]：{unit.name}，需了解：{required_fields}        │
│    [已知]：{covered_fields_in_this_unit}                   │
│    [近3轮]：{unit_history[-3:]}                            │
│    [用户刚说]：{user_text}（情绪：{emotion}）               │
│    [可选追问]：{follow_up_options}                         │
│    输出 JSON，无其他内容                                    │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  LLM-2：提取（Qwen-plus）                                   │
│                                                            │
│  输入：用户文本 + 全量字段定义表                             │
│  输出：{field_name: value, ...}（仅出现的字段）              │
│  运行时机：后台，用户说话同时运行                            │
│  不影响响应延迟                                             │
│                                                            │
│  关键设计：全局提取（不限当前单元）                          │
│    → 用户回答单元5时提到离婚，单元180（婚姻）自动标记         │
└────────────────────────────────────────────────────────────┘
```

### 5.5 Python 状态机

```python
class InterviewStateMachine:

    def transition(self, state: InterviewState,
                   decision: TurnDecision) -> InterviewState:

        if decision.action == "empathy_first":
            # 共情后重复当前问题
            return state  # 不变，下一轮继续当前问题

        if decision.action == "follow_up":
            state.current_question_id = decision.next_question_id
            return state

        if decision.action in ("next_question", "next_unit"):
            state.completed_unit_ids.append(state.current_unit_id)
            state.unit_history = []  # 清空当前单元历史

            # 找下一个未完成且应该问的单元
            next_unit_id = self._get_next_unit(state)

            if next_unit_id is None:
                state.status = InterviewStatus.DIAGNOSING
                return state

            state.current_unit_id = next_unit_id
            state.current_question_id = self._get_first_unanswered_q(
                next_unit_id, state.extracted
            )
            return state

    def _get_next_unit(self, state):
        for uid in state.units_queue:
            if uid in state.completed_unit_ids:
                continue
            unit = question_bank.get_unit(uid)
            # 检查是否可跳过（信息已知）
            if self._should_skip(unit, state.extracted):
                state.completed_unit_ids.append(uid)
                continue
            return uid
        return None

    def _should_skip(self, unit, extracted):
        return all(f in extracted for f in unit.required_fields)
```

### 5.6 诊断与报告层

```
诊断引擎（纯规则，无 LLM）：

  input:  state.extracted（全量提取结果）
  output: RiskReport

  流程：
    for unit in all_units:
        score = scoring_rule.evaluate(extracted)
        # 示例规则：
        # if 月收入 < 5000 and 固定支出 > 月收入*0.8:
        #     income_risk = "HIGH"

  结果结构：
    {
      "unit_scores":      {unit_id: score},        # 260个
      "dimension_scores": {                         # 6个维度汇总
          "收入稳定性": 72,
          "资产负债":   45,
          "保障缺口":   88,
          ...
      },
      "overall_risk":     "中高风险",
      "top_risks":        [top 5 风险项],
      "recommendations":  [...]
    }

报告生成：
  Jinja2 HTML 模板 → WeasyPrint → PDF
  耗时：3-5秒
  包含：雷达图（matplotlib 预渲染）+ 文字分析 + 建议清单
```

---

## 六、完整用户体验流程

```
用户打开浏览器
    ↓
[前端] 请求麦克风权限 → 建立 LiveKit WebRTC 连接
    ↓
[系统] 播放欢迎语（预生成）
    「您好，我是您的财富风险诊断助手。
      接下来我们通过对话了解您的财务状况，
      大约需要 45-60 分钟，您随时可以打断我提问。」
    ↓
    ┌──────────────────────────────────────────────┐
    │             访谈循环                          │
    │                                              │
    │  系统：播放问题（0ms，预生成音频）              │
    │    ↓                                         │
    │  用户：语音回答（3-30秒）                      │
    │    ↓ 同时后台：LLM判断 + LLM提取              │
    │  VAD：检测说完（300ms）                       │
    │    ↓                                         │
    │  情绪检测：                                   │
    │    → sad/anxious → 播放共情语，暂停1轮         │
    │    → neutral    → 直接播放下一句               │
    │    ↓                                         │
    │  响应时间：200-500ms（用户无感知等待）          │
    │    ↓                                         │
    │  决策：                                       │
    │    → 追问  → 播放追问题                       │
    │    → 充分  → 单元过渡语 + 下一单元第一题        │
    │    → 已知  → 静默跳过，直接下一单元            │
    │                                              │
    │  进度提示（每10个单元）：                      │
    │    「我们已经了解了收入和资产情况，             │
    │      接下来聊聊您的保障规划……」                │
    └──────────────────────────────────────────────┘
    ↓
[系统] 「感谢您的配合，正在为您生成诊断报告……」
    ↓
规则评分（< 1秒） → PDF 生成（3-5秒）
    ↓
[前端] 展示报告摘要 + PDF 下载链接
```

---

## 七、时延设计目标

| 场景 | 目标延迟 | 实现方式 |
|------|---------|---------|
| 用户说完 → 系统开口 | < 500ms | 预生成音频 + Qwen-turbo |
| 系统开口 → 第一个音节 | < 100ms | 音频已在缓冲区 |
| 用户打断 → 系统停播 | < 50ms | Barge-in + VAD |
| 网络抖动下的流畅度 | 有保障 | LiveKit WebRTC 抖动缓冲 |
| ASR 识别延迟（300ms静音后） | 300-500ms | SenseVoice 流式 |
| LLM 判断延迟 | 200-400ms | Qwen-turbo |
| LLM 提取延迟（后台） | 500-1500ms | 不影响主路径 |
| **用户感知响应延迟（综合）** | **300-600ms** | **自然对话级别** |

---

## 八、异常处理

| 异常场景 | 处理方式 |
|---------|---------|
| ASR 置信度低（< 0.6） | 播放「不好意思，没听清楚，请再说一遍」 |
| 用户静音 > 10秒 | 播放「您还在吗？我们可以继续……」 |
| 用户静音 > 30秒 | 暂停会话，发送续接链接 |
| 网络断开 | Redis 保存当前 state，支持续接 |
| LLM 超时（> 2s） | 降级：按预设顺序取下一题，不等 LLM |
| LLM 返回格式错误 | 重试一次，失败则按预设逻辑推进 |
| 用户说「跳过/不想说」 | 标记字段为「拒答」，移入下一单元 |

---

## 九、部署架构

```
┌────────────────────────────────────────────────────┐
│                   生产环境                          │
│                                                    │
│  Nginx（SSL终止 + 静态文件）                        │
│       ↓                                            │
│  FastAPI（uvicorn，4进程）                          │
│       ↓                                            │
│  LiveKit Server（独立部署 or 云服务）                │
│                                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ FunASR   │  │CosyVoice │  │  PostgreSQL+Redis │ │
│  │ Server   │  │ Server   │  │                  │ │
│  │（本地GPU）│  │（本地GPU）│  │                  │ │
│  └──────────┘  └──────────┘  └──────────────────┘ │
│                                                    │
│  GPU 要求：FunASR+CosyVoice 共享，最低 8GB VRAM    │
└────────────────────────────────────────────────────┘
```

---

## 十、安全与合规

| 项目 | 措施 |
|------|------|
| 录音告知同意 | 访谈开始前展示同意弹窗，记录时间戳 |
| 数据加密传输 | LiveKit DTLS（强制）+ HTTPS |
| 静态数据加密 | PostgreSQL 字段加密（敏感字段） |
| 音频存储 | 默认不存储原始音频；可配置存 30 天 |
| 访问控制 | 每个会话独立 token，服务端验证 |
| PIPL 合规 | 数据本地化（国内部署），支持用户数据删除 |
| 报告访问控制 | PDF 链接带时效 token，30 分钟过期 |

---

## 十一、关键决策记录（ADR）

| 决策 | 选择 | 放弃的选项 | 理由 |
|------|------|-----------|------|
| 音频传输 | LiveKit WebRTC | WebSocket | 内置 AEC，更低延迟，生产级稳定性 |
| ASR | FunASR SenseVoice | Whisper | 中文最优，含情绪，流式支持 |
| TTS | CosyVoice 2 预生成 | 实时合成 | 访谈题目固定，预生成消除延迟 |
| LLM 框架 | 直接调 SDK | LiteLLM/LangChain | 减少依赖，减少延迟，更透明 |
| 状态机 | 纯 Python | LangGraph | 简单场景不需要图框架 |
| 诊断 | 规则引擎 | LLM评分 | 可解释性强，零延迟，可审计 |
| 判断/提取 | 分离两个 LLM | 合并一个调用 | 判断需要快，提取不阻塞主路径 |

---

## 十二、访谈分层设计（待细化）

```
当前问题：260 个单元平铺，按序访问
目标：    分层触发，45-60 分钟完成

分层方案：

  核心层（~30单元，必问）
    └── 基础人口信息、收入、资产概览、家庭结构
        → 为触发条件提供基础字段

  扩展层（~230单元，条件触发）
    └── 由核心层提取结果动态解锁
        示例触发条件：
          "已婚"                    → 解锁：配偶收入、婚姻风险单元
          "有子女"                  → 解锁：子女教育规划单元
          "有房贷"                  → 解锁：房贷压力、还款能力单元
          "有企业"                  → 解锁：企业财务风险、股权单元
          "月收入 > 5万"            → 解锁：高净值资产配置单元

待定策略：
  1. 核心层30个单元的具体选择（领域专家评审）
  2. 触发条件粒度：单元级 or 问题级？
  3. 时间超限时的智能收尾策略
  4. 同优先级单元的排序规则
```

---

*文档版本 v1.0 | 生成日期 2026-03-02*
