# 财富风险诊断访谈系统 — 开发计划 v2.0

> 基于架构设计文档 v1.0 | 制定日期：2026-03-02

---

## 总体策略

**原则：垂直切片，逐层可运行**

每个阶段结束后系统必须可端到端运行（哪怕功能残缺），不做只能在最后才能联调的横向切割。

```
阶段划分：

  Phase 0  项目骨架 & 基础设施（1-2天）
  Phase 1  语音管道：LiveKit + VAD + ASR（2-3天）
  Phase 2  TTS 预生成 & 题库数据层（2天）
  Phase 3  访谈引擎核心：状态机 + LLM 判断（3-4天）
  Phase 4  提取循环 & 分层触发（2-3天）
  Phase 5  诊断引擎 & 报告生成（2-3天）
  Phase 6  前端（HTML/JS + LiveKit Client）（2天）
  Phase 7  持久化：PostgreSQL + Redis（1-2天）
  Phase 8  异常处理 & 健壮性（1-2天）
  Phase 9  集成测试 & 延迟调优（2天）
  Phase 10 部署 & 上线（1-2天）

  总计：~20-25 个工作日
```

---

## Phase 0 — 项目骨架 & 基础设施

### 目标
可运行的空壳：FastAPI 启动、LiveKit 连通、依赖全部可导入。

### 目录结构

```
wealth-risk-interview/
├── app/
│   ├── main.py               # FastAPI 入口
│   ├── config.py             # 环境变量配置
│   ├── models/
│   │   ├── interview.py      # InterviewState, InterviewStatus
│   │   ├── question_bank.py  # RiskUnit, Question, QuestionTier
│   │   └── llm.py            # TurnDecision
│   ├── pipeline/
│   │   ├── vad.py            # Silero VAD 封装
│   │   ├── asr.py            # FunASR SenseVoice 封装
│   │   ├── tts.py            # CosyVoice 2 封装
│   │   └── livekit_agent.py  # LiveKit 音频收发
│   ├── engine/
│   │   ├── state_machine.py  # InterviewStateMachine
│   │   ├── decision.py       # LLM 判断（Qwen-turbo）
│   │   ├── extraction.py     # LLM 提取（Qwen-plus）
│   │   └── trigger.py        # 扩展层触发条件评估
│   ├── diagnosis/
│   │   ├── scorer.py         # 规则评分引擎
│   │   └── report.py         # Jinja2 + WeasyPrint PDF
│   ├── storage/
│   │   ├── postgres.py       # 会话持久化
│   │   └── redis_cache.py    # 音频缓存 & 会话状态
│   ├── api/
│   │   └── routes.py         # HTTP 端点
│   └── static/
│       └── index.html        # 前端单页
├── data/
│   ├── questions/            # 题库 YAML/JSON
│   └── audio_cache/          # 预生成 TTS 音频
├── templates/
│   └── report.html           # Jinja2 报告模板
├── tests/
│   ├── test_pipeline.py
│   ├── test_engine.py
│   └── test_diagnosis.py
├── scripts/
│   └── pregenerate_tts.py    # 一次性预生成所有题目音频
├── pyproject.toml
├── .env.example
└── docker-compose.yml
```

### 依赖清单（pyproject.toml）

```toml
[tool.poetry.dependencies]
python = "^3.11"

# 服务框架
fastapi = "^0.115"
uvicorn = {extras = ["standard"], version = "^0.30"}

# WebRTC
livekit = "^0.11"
livekit-agents = "^0.8"

# 语音处理
torch = "^2.2"
torchaudio = "^2.2"
silero-vad = "^5.0"          # VAD

# ASR（FunASR）
funasr = "^1.1"
modelscope = "^1.15"

# TTS（CosyVoice 2）
cosyvoice = {git = "https://github.com/FunAudioLLM/CosyVoice"}

# LLM
dashscope = "^1.20"           # Qwen API

# 数据模型
pydantic = "^2.7"

# 存储
asyncpg = "^0.29"
redis = {extras = ["asyncio"], version = "^5.0"}

# 报告
jinja2 = "^3.1"
weasyprint = "^62"
matplotlib = "^3.9"

# 工具
python-dotenv = "^1.0"
pyyaml = "^6.0"
```

### 验收标准
- [ ] `uvicorn app.main:app` 启动无报错
- [ ] `GET /health` 返回 200
- [ ] LiveKit Room 可建立连接（本地 LiveKit dev server）
- [ ] 所有依赖 `import` 无报错

---

## Phase 1 — 语音管道

### 目标
麦克风输入 → VAD → ASR → 文字输出；系统文字 → TTS → 播放。
实现 Barge-in。

### 1.1 Silero VAD 封装

```python
# app/pipeline/vad.py
class SileroVAD:
    def __init__(self, sample_rate=16000, threshold=0.5):
        self.model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad')
        self.sample_rate = sample_rate
        self.threshold = threshold
        self._speech_started = False

    def process_chunk(self, pcm_chunk: bytes) -> tuple[bool, bool]:
        """
        返回 (speech_start, speech_end)
        每次喂入 20ms PCM（320 samples @ 16kHz）
        """
        tensor = torch.frombuffer(pcm_chunk, dtype=torch.int16).float() / 32768.0
        prob = self.model(tensor, self.sample_rate).item()

        speech_start = False
        speech_end = False

        if prob > self.threshold and not self._speech_started:
            self._speech_started = True
            speech_start = True
        elif prob < self.threshold and self._speech_started:
            self._speech_started = False
            speech_end = True

        return speech_start, speech_end
```

### 1.2 FunASR SenseVoice 封装

```python
# app/pipeline/asr.py
class SenseVoiceASR:
    """
    流式接口封装
    输出：(text: str, emotion: str, confidence: float)
    emotion ∈ {"neutral", "sad", "anxious", "happy", "angry"}
    """
    def __init__(self):
        from funasr import AutoModel
        self.model = AutoModel(
            model="iic/SenseVoiceSmall",
            trust_remote_code=True,
            device="cuda"
        )
        self._buffer = []

    def feed(self, pcm_chunk: bytes):
        self._buffer.append(pcm_chunk)

    async def finalize(self) -> tuple[str, str, float]:
        audio = b"".join(self._buffer)
        self._buffer = []
        result = self.model.generate(
            input=audio,
            language="zh",
            use_itn=True,
            batch_size_s=60
        )
        text = result[0]["text"]
        emotion = result[0].get("emotion", "neutral")
        confidence = result[0].get("confidence", 1.0)
        return text, emotion, confidence
```

### 1.3 Barge-in 控制器

```python
# app/pipeline/livekit_agent.py
class BargeInController:
    def __init__(self):
        self._barge_in = asyncio.Event()

    def signal_barge_in(self):
        self._barge_in.set()

    async def stream_audio(self, audio_bytes: bytes,
                           room: rtc.Room, source: rtc.AudioSource):
        CHUNK = 960  # 20ms @ 48kHz
        self._barge_in.clear()
        for i in range(0, len(audio_bytes), CHUNK):
            if self._barge_in.is_set():
                self._barge_in.clear()
                return  # 立即停止
            frame = rtc.AudioFrame(
                data=audio_bytes[i:i+CHUNK],
                sample_rate=48000,
                num_channels=1,
                samples_per_channel=CHUNK // 2
            )
            await source.capture_frame(frame)
            await asyncio.sleep(0.02)
```

### 验收标准
- [ ] 对着麦克风说话，控制台打印识别文字 + 情绪
- [ ] 播放音频期间说话，音频立即停止（< 100ms）
- [ ] ASR 置信度字段正常输出
- [ ] emotion 字段在说悲伤内容时输出 `sad`

---

## Phase 2 — TTS 预生成 & 题库数据层

### 目标
所有题目音频提前生成完毕；题库以结构化格式加载到内存。

### 2.1 题库格式（YAML）

```yaml
# data/questions/unit_01_income.yaml
unit_id: "unit_01"
name: "收入来源与稳定性"
tier: "core"
trigger_condition: null
required_fields:
  - "月收入"
  - "收入来源类型"
  - "收入稳定性"
scoring_rule:
  type: "income_stability"
  weights:
    月收入: 0.4
    收入稳定性: 0.6

questions:
  - question_id: "unit_01_q1"
    text: "您好，我们先聊聊您的收入情况。请问您目前主要的收入来源是什么？"
    audio_path: "data/audio_cache/unit_01_q1.wav"
    covers_fields: ["收入来源类型"]
    follow_up_ids: ["unit_01_q1_f1", "unit_01_q1_f2"]

  - question_id: "unit_01_q1_f1"
    text: "您的工资收入大概是每月多少？"
    audio_path: "data/audio_cache/unit_01_q1_f1.wav"
    covers_fields: ["月收入"]
    follow_up_ids: []

  - question_id: "unit_01_q1_f2"
    text: "这份收入稳定吗？比如会有月份差异比较大的情况吗？"
    audio_path: "data/audio_cache/unit_01_q1_f2.wav"
    covers_fields: ["收入稳定性"]
    follow_up_ids: []
```

### 2.2 预生成脚本

```python
# scripts/pregenerate_tts.py
"""
一次性运行，为所有题目生成 WAV 文件
运行：python scripts/pregenerate_tts.py
"""
import asyncio, yaml, pathlib
from app.pipeline.tts import CosyVoiceTTS

async def main():
    tts = CosyVoiceTTS()
    questions_dir = pathlib.Path("data/questions")
    audio_dir = pathlib.Path("data/audio_cache")
    audio_dir.mkdir(exist_ok=True)

    for yaml_file in questions_dir.glob("*.yaml"):
        unit = yaml.safe_load(yaml_file.read_text())
        for q in unit["questions"]:
            audio_path = pathlib.Path(q["audio_path"])
            if audio_path.exists():
                continue  # 已生成，跳过
            audio_bytes = await tts.synthesize(q["text"])
            audio_path.write_bytes(audio_bytes)
            print(f"生成: {q['question_id']}")

    # 同时生成：欢迎语、共情语、过渡语、结束语
    special_texts = {
        "welcome.wav": "您好，我是您的财富风险诊断助手……",
        "empathy_sad.wav": "听起来这段时间您承受了不少压力，没关系，我们慢慢来……",
        "empathy_anxious.wav": "我理解这些话题可能让您感到有些担忧……",
        "transition.wav": "好的，我们已经了解了这方面的情况，接下来……",
        "farewell.wav": "感谢您的配合，正在为您生成诊断报告……",
    }
    for filename, text in special_texts.items():
        path = audio_dir / filename
        if not path.exists():
            path.write_bytes(await tts.synthesize(text))

asyncio.run(main())
```

### 2.3 题库内存加载

```python
# app/models/question_bank.py
class QuestionBank:
    """单例，启动时加载，所有查询 O(1)"""

    def __init__(self):
        self._units: dict[str, RiskUnit] = {}
        self._questions: dict[str, Question] = {}
        self._audio_cache: dict[str, bytes] = {}  # 全部预读入内存

    def load(self, questions_dir: str):
        for yaml_file in pathlib.Path(questions_dir).glob("*.yaml"):
            data = yaml.safe_load(yaml_file.read_text())
            unit = RiskUnit(**data)
            self._units[unit.unit_id] = unit
            for q in unit.questions:
                self._questions[q.question_id] = q
                self._audio_cache[q.question_id] = pathlib.Path(q.audio_path).read_bytes()

    def get_audio(self, question_id: str) -> bytes:
        return self._audio_cache[question_id]  # 0ms

    def get_unit(self, unit_id: str) -> RiskUnit:
        return self._units[unit_id]

    def get_core_units(self) -> list[RiskUnit]:
        return [u for u in self._units.values() if u.tier == QuestionTier.CORE]

    def get_extended_units(self) -> list[RiskUnit]:
        return [u for u in self._units.values() if u.tier == QuestionTier.EXTENDED]

question_bank = QuestionBank()  # 全局单例
```

### 验收标准
- [ ] `python scripts/pregenerate_tts.py` 成功生成所有音频文件
- [ ] 题库加载后 `question_bank.get_audio("unit_01_q1")` 返回有效音频
- [ ] 启动日志打印题库大小：`Loaded X units, Y questions`
- [ ] 音频文件总大小在预期范围内（估算：1000题 × 5秒 × 32KB/s ≈ 160MB）

---

## Phase 3 — 访谈引擎核心

### 目标
完整的一轮对话可运行：系统问 → 用户答 → LLM 判断 → 下一问。

### 3.1 LLM 判断（Qwen-turbo）

```python
# app/engine/decision.py
JUDGE_SYSTEM_PROMPT = """你是一个访谈决策助手。
根据当前访谈单元的目标、已知信息和用户回答，决定下一步行动。
只输出 JSON，不输出其他内容。"""

JUDGE_USER_TEMPLATE = """
当前单元：{unit_name}
需要了解的字段：{required_fields}
本单元已知字段：{known_fields}
最近3轮对话：
{unit_history}
用户刚说："{user_text}"（情绪：{emotion}）
可选追问题：{follow_up_options}

输出格式：
{{
  "action": "follow_up" | "next_question" | "next_unit" | "empathy_first",
  "next_question_id": "...",
  "unit_complete": true | false,
  "reason": "..."
}}

决策规则：
- 若 required_fields 均已知 → next_unit，unit_complete=true
- 若用户提供了部分信息但还有必要字段未覆盖 → follow_up
- 若情绪为 sad/anxious 且本轮未共情 → empathy_first
- 若用户拒绝回答 → next_unit，在 reason 中标注「拒答」
"""

async def llm_judge(state: InterviewState,
                    user_text: str,
                    emotion: str) -> TurnDecision:
    unit = question_bank.get_unit(state.current_unit_id)
    known_fields = [f for f in unit.required_fields
                    if f in state.extracted]
    current_q = question_bank._questions[state.current_question_id]
    follow_up_options = [
        {"id": fid, "text": question_bank._questions[fid].text}
        for fid in current_q.follow_up_ids
    ]

    prompt = JUDGE_USER_TEMPLATE.format(
        unit_name=unit.name,
        required_fields=unit.required_fields,
        known_fields=known_fields,
        unit_history=format_history(state.unit_history[-3:]),
        user_text=user_text,
        emotion=emotion,
        follow_up_options=follow_up_options,
    )

    import dashscope
    response = dashscope.Generation.call(
        model="qwen-turbo",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        result_format="message",
    )
    raw = response.output.choices[0].message.content
    return TurnDecision(**json.loads(raw))
```

### 3.2 Python 状态机

```python
# app/engine/state_machine.py
class InterviewStateMachine:

    def __init__(self, question_bank: QuestionBank):
        self.qb = question_bank

    def init_state(self, session_id: str) -> InterviewState:
        """初始化：构建核心层队列"""
        core_unit_ids = [u.unit_id for u in self.qb.get_core_units()]
        first_unit = core_unit_ids[0]
        first_q = self.qb.get_unit(first_unit).questions[0].question_id

        return InterviewState(
            session_id=session_id,
            status=InterviewStatus.GREETING,
            units_queue=core_unit_ids,
            current_unit_id=first_unit,
            current_question_id=first_q,
            completed_unit_ids=[],
            extracted={},
            unit_history=[],
            full_history=[],
            emotion_log=[],
        )

    def transition(self, state: InterviewState,
                   decision: TurnDecision,
                   user_text: str,
                   emotion: str) -> InterviewState:

        # 记录历史
        state.unit_history.append({"role": "user", "content": user_text})
        state.full_history.append({"role": "user", "content": user_text,
                                   "emotion": emotion, "unit": state.current_unit_id})
        state.emotion_log.append({"unit": state.current_unit_id,
                                   "emotion": emotion, "text": user_text})

        if decision.action == "empathy_first":
            return state  # 保持当前问题不变

        if decision.action == "follow_up":
            state.current_question_id = decision.next_question_id
            return state

        if decision.action in ("next_question", "next_unit"):
            if decision.unit_complete:
                state.completed_unit_ids.append(state.current_unit_id)
                state.unit_history = []

            next_unit_id = self._get_next_unit(state)

            if next_unit_id is None:
                state.status = InterviewStatus.DIAGNOSING
                return state

            state.current_unit_id = next_unit_id
            state.current_question_id = self._get_entry_question(
                next_unit_id, state.extracted
            )
            return state

        return state

    def _get_next_unit(self, state: InterviewState) -> str | None:
        for uid in state.units_queue:
            if uid in state.completed_unit_ids:
                continue
            unit = self.qb.get_unit(uid)
            if self._should_skip(unit, state.extracted):
                state.completed_unit_ids.append(uid)
                continue
            return uid
        return None

    def _should_skip(self, unit: RiskUnit, extracted: dict) -> bool:
        """所有必要字段已从其他单元提取到 → 跳过此单元"""
        return all(f in extracted for f in unit.required_fields)

    def _get_entry_question(self, unit_id: str, extracted: dict) -> str:
        """选择单元第一个尚未被覆盖的问题"""
        unit = self.qb.get_unit(unit_id)
        for q in unit.questions:
            uncovered = [f for f in q.covers_fields if f not in extracted]
            if uncovered:
                return q.question_id
        return unit.questions[0].question_id
```

### 3.3 主访谈循环

```python
# app/pipeline/livekit_agent.py（访谈主循环）

async def run_interview(session_id: str, room: rtc.Room):
    """三协程并发运行"""
    state = state_machine.init_state(session_id)
    asr_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
    extract_queue: asyncio.Queue[dict] = asyncio.Queue()
    audio_buffer: asyncio.Queue[bytes] = asyncio.Queue()
    barge_in = BargeInController()

    # 播放欢迎语
    await audio_buffer.put(
        question_bank.get_special_audio("welcome.wav")
    )

    await asyncio.gather(
        voice_loop(room, asr_queue, barge_in),
        decision_loop(state, asr_queue, extract_queue, audio_buffer),
        playback_loop(room, audio_buffer, barge_in),
        extraction_loop(state, extract_queue),
    )
```

### 验收标准
- [ ] 完整走完 3 个核心单元（命令行 mock 音频输入）
- [ ] 状态机正确转移：greeting → interviewing → diagnosing
- [ ] LLM 判断返回有效 JSON，解析不报错
- [ ] 判断延迟 P95 < 500ms（Qwen-turbo）
- [ ] 用户拒绝回答时系统正确推进到下一单元

---

## Phase 4 — 提取循环 & 分层触发

### 目标
后台提取用户回答中的结构化字段；自动解锁扩展层单元。

### 4.1 LLM 提取（Qwen-plus）

```python
# app/engine/extraction.py
EXTRACT_SYSTEM = """你是信息提取助手。
从用户的口语化表达中提取结构化字段。
只提取明确提及的字段，不要推断，不要补全。
输出 JSON。"""

# 字段定义表（供 LLM 参考）
FIELD_DEFINITIONS = {
    "月收入":       "数字，单位元，月收入",
    "收入来源类型":  "字符串：工资/个体/投资/多元",
    "婚姻状态":     "字符串：未婚/已婚/离婚/丧偶",
    "子女数量":     "数字",
    "月固定支出":   "数字，单位元",
    # ... 全量字段定义
}

async def llm_extract(text: str) -> dict[str, Any]:
    prompt = f"""
字段定义：{json.dumps(FIELD_DEFINITIONS, ensure_ascii=False)}
用户说："{text}"
提取所有明确提及的字段，输出 JSON。
若某字段未提及，不要包含在输出中。
"""
    response = dashscope.Generation.call(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        result_format="message",
    )
    raw = response.output.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}  # 提取失败，不影响主流程
```

### 4.2 触发条件评估

```python
# app/engine/trigger.py
class TriggerEvaluator:
    """
    评估哪些扩展单元应当被解锁
    触发条件语法（YAML 中定义）：
      "已婚"                           → extracted["婚姻状态"] == "已婚"
      "有子女"                         → extracted.get("子女数量", 0) > 0
      "有房贷"                         → extracted.get("有房贷") == True
      "月收入 > 50000"                  → extracted.get("月收入", 0) > 50000
      "已婚 AND 有子女"                 → 组合条件
    """

    def evaluate(self, condition: str, extracted: dict) -> bool:
        if condition is None:
            return True
        # 简单表达式解析（AND/OR/比较）
        return self._eval_expr(condition, extracted)

    def _eval_expr(self, expr: str, extracted: dict) -> bool:
        if " AND " in expr:
            return all(self._eval_single(e.strip(), extracted)
                       for e in expr.split(" AND "))
        if " OR " in expr:
            return any(self._eval_single(e.strip(), extracted)
                       for e in expr.split(" OR "))
        return self._eval_single(expr, extracted)

    def _eval_single(self, expr: str, extracted: dict) -> bool:
        SIMPLE_CONDITIONS = {
            "已婚":   lambda e: e.get("婚姻状态") == "已婚",
            "有子女":  lambda e: int(e.get("子女数量", 0)) > 0,
            "有房贷":  lambda e: e.get("有房贷") is True,
            "有企业":  lambda e: e.get("有企业") is True,
        }
        # 数值比较：月收入 > 50000
        import re
        m = re.match(r"(\S+)\s*(>|<|>=|<=|==)\s*(\S+)", expr)
        if m:
            field, op, val = m.groups()
            actual = extracted.get(field)
            if actual is None:
                return False
            return eval(f"{actual} {op} {val}")  # noqa（受控表达式）

        fn = SIMPLE_CONDITIONS.get(expr)
        return fn(extracted) if fn else False


def evaluate_triggers(extracted: dict, all_units: list[RiskUnit],
                      already_queued: list[str]) -> list[RiskUnit]:
    evaluator = TriggerEvaluator()
    newly_unlocked = []
    for unit in all_units:
        if unit.tier != QuestionTier.EXTENDED:
            continue
        if unit.unit_id in already_queued:
            continue
        if evaluator.evaluate(unit.trigger_condition, extracted):
            newly_unlocked.append(unit)
    return newly_unlocked
```

### 4.3 提取协程

```python
async def extraction_loop(state: InterviewState,
                          extract_queue: asyncio.Queue):
    evaluator = TriggerEvaluator()
    all_extended = question_bank.get_extended_units()

    while True:
        task = await extract_queue.get()
        try:
            result = await llm_extract(task["text"])
            state.extracted.update(result)

            # 检查新解锁的扩展单元
            newly_unlocked = evaluate_triggers(
                state.extracted, all_extended, state.units_queue
            )
            for unit in newly_unlocked:
                # 插入当前位置之后（不打乱已排队的核心层顺序）
                insert_at = len(state.completed_unit_ids) + 1
                state.units_queue.insert(insert_at, unit.unit_id)

        except Exception:
            pass  # 提取失败不影响主流程
        finally:
            extract_queue.task_done()
```

### 验收标准
- [ ] 用户说「我有两个孩子」→ `extracted["子女数量"] == 2`
- [ ] 用户说「我和老婆结婚十年了」→ `extracted["婚姻状态"] == "已婚"`
- [ ] 解锁条件满足后，对应扩展单元出现在 `units_queue` 中
- [ ] 提取失败不影响主循环继续运行
- [ ] 跨单元提取：在单元5提到的离婚，单元180对应字段已标记

---

## Phase 5 — 诊断引擎 & 报告生成

### 目标
访谈结束后 < 10 秒内生成 PDF 报告。

### 5.1 规则评分引擎

```python
# app/diagnosis/scorer.py
class RiskScorer:

    def score_all(self, extracted: dict) -> RiskReport:
        unit_scores = {}
        for unit in question_bank._units.values():
            score = self._score_unit(unit, extracted)
            unit_scores[unit.unit_id] = score

        dimension_scores = self._aggregate_dimensions(unit_scores)
        overall = self._calc_overall(dimension_scores)
        top_risks = self._find_top_risks(unit_scores, extracted)
        recommendations = self._generate_recommendations(top_risks, extracted)

        return RiskReport(
            unit_scores=unit_scores,
            dimension_scores=dimension_scores,
            overall_risk=overall,
            top_risks=top_risks,
            recommendations=recommendations,
        )

    def _score_unit(self, unit: RiskUnit, extracted: dict) -> float:
        rule = unit.scoring_rule
        score_type = rule.get("type")

        if score_type == "income_stability":
            income = extracted.get("月收入", 0)
            expenses = extracted.get("月固定支出", 0)
            if income == 0:
                return 0.0
            ratio = expenses / income
            base = min(income / 20000, 1.0)  # 归一化
            stability_penalty = ratio if ratio > 0.7 else 0
            return max(0.0, base - stability_penalty)

        # ... 其他规则类型

        return 0.5  # 默认中等风险（数据缺失时）
```

### 5.2 雷达图生成

```python
# app/diagnosis/report.py
def generate_radar_chart(dimension_scores: dict) -> str:
    """返回 base64 PNG，嵌入 HTML"""
    import matplotlib.pyplot as plt
    import numpy as np, io, base64

    labels = list(dimension_scores.keys())
    values = list(dimension_scores.values())
    N = len(labels)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6),
                            subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 100)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()
```

### 5.3 PDF 生成

```python
async def generate_report(state: InterviewState,
                           report: RiskReport) -> bytes:
    radar_b64 = generate_radar_chart(report.dimension_scores)

    html = jinja2.Environment(
        loader=jinja2.FileSystemLoader("templates")
    ).get_template("report.html").render(
        client_name="受访客户",
        date=datetime.now().strftime("%Y年%m月%d日"),
        radar_chart=radar_b64,
        dimension_scores=report.dimension_scores,
        overall_risk=report.overall_risk,
        top_risks=report.top_risks,
        recommendations=report.recommendations,
        emotion_summary=summarize_emotions(state.emotion_log),
    )

    pdf_bytes = weasyprint.HTML(string=html).write_pdf()
    return pdf_bytes
```

### 验收标准
- [ ] 给定模拟 `extracted` 字典，评分引擎输出所有维度分数
- [ ] 雷达图正确渲染（6个维度）
- [ ] PDF 生成时间 < 10 秒
- [ ] PDF 包含：封面、雷达图、Top 5 风险、建议清单
- [ ] 拒答字段在报告中标注「未提供」而非异常

---

## Phase 6 — 前端

### 目标
浏览器端可完整完成一次访谈，包含进度显示和情绪 UI 反馈。

### 6.1 单页 HTML 结构

```html
<!-- app/static/index.html 核心结构 -->
<div id="app">
  <!-- 波形动画（系统说话时激活） -->
  <div id="waveform" class="idle">
    <canvas id="wave-canvas"></canvas>
    <span id="status-text">准备中…</span>
  </div>

  <!-- 字幕区 -->
  <div id="subtitle"></div>

  <!-- 进度条 -->
  <div id="progress-bar">
    <span id="progress-text">第 0 / 约 30 个话题</span>
    <div id="progress-fill" style="width: 0%"></div>
  </div>

  <!-- 麦克风状态 -->
  <div id="mic-status">🎤 正在聆听</div>
</div>
```

### 6.2 LiveKit JS 集成

```javascript
// DataChannel 事件处理
room.on(RoomEvent.DataReceived, (payload, participant) => {
  const msg = JSON.parse(new TextDecoder().decode(payload));

  switch (msg.type) {
    case "subtitle":
      document.getElementById("subtitle").textContent = msg.text;
      break;
    case "progress":
      const pct = (msg.current / msg.total * 100).toFixed(0);
      document.getElementById("progress-fill").style.width = pct + "%";
      document.getElementById("progress-text").textContent =
        `第 ${msg.current} / 约 ${msg.total} 个话题`;
      break;
    case "empathy":
      document.getElementById("app").classList.add("empathy-mode");
      setTimeout(() => document.getElementById("app")
        .classList.remove("empathy-mode"), 3000);
      break;
    case "done":
      window.location.href = msg.report_url;
      break;
  }
});
```

### 验收标准
- [ ] 浏览器打开即可开始访谈，无需安装任何软件
- [ ] 进度条随访谈推进实时更新
- [ ] 系统说话时波形动画激活
- [ ] 用户打断时波形状态正确切换
- [ ] 情绪触发时 UI 颜色/样式变化
- [ ] 手机浏览器（Safari/Chrome）可用

---

## Phase 7 — 持久化

### 目标
会话可续接；音频缓存可热加载。

### 7.1 PostgreSQL（asyncpg）

```sql
-- 会话表
CREATE TABLE sessions (
    session_id    UUID PRIMARY KEY,
    created_at    TIMESTAMPTZ DEFAULT now(),
    updated_at    TIMESTAMPTZ DEFAULT now(),
    status        TEXT NOT NULL,
    state_json    JSONB NOT NULL,    -- InterviewState 序列化
    report_pdf    BYTEA,            -- 生成后存储
    consent_at    TIMESTAMPTZ       -- 知情同意时间戳
);

-- 提取结果表（按单元拆分，方便分析）
CREATE TABLE extractions (
    id            SERIAL PRIMARY KEY,
    session_id    UUID REFERENCES sessions,
    unit_id       TEXT,
    field_name    TEXT,
    field_value   JSONB,
    extracted_at  TIMESTAMPTZ DEFAULT now()
);
```

### 7.2 Redis

```python
# app/storage/redis_cache.py
class RedisCache:
    async def save_state(self, session_id: str, state: InterviewState):
        """断线保护：每轮对话后保存"""
        await redis.set(
            f"session:{session_id}:state",
            state.model_dump_json(),
            ex=3600  # 1小时过期
        )

    async def load_state(self, session_id: str) -> InterviewState | None:
        raw = await redis.get(f"session:{session_id}:state")
        if raw:
            return InterviewState.model_validate_json(raw)
        return None
```

### 验收标准
- [ ] 访谈中途断开，重新连接后从断点继续
- [ ] 报告 PDF 存入 PostgreSQL，可通过 API 下载
- [ ] Redis 中会话状态 TTL 正常（1小时）

---

## Phase 8 — 异常处理 & 健壮性

### 目标
任何单点故障不崩溃系统。

### 异常处理矩阵

| 场景 | 检测方式 | 处理 |
|------|---------|------|
| ASR 置信度 < 0.6 | confidence 字段 | 播放「没听清楚」，重复当前问题 |
| 用户静音 > 10s | asyncio.wait_for 超时 | 播放「您还在吗？」 |
| 用户静音 > 30s | 二次超时 | 暂停，发续接链接到前端 |
| LLM 超时 > 2s | asyncio.wait_for | 降级：顺序取下一题 |
| LLM JSON 解析失败 | try/except | 重试1次，失败则顺序推进 |
| 音频文件缺失 | 启动时检查 | 阻断启动，报错明确 |
| 用户说「跳过/不说」 | LLM 判断识别 | 字段标记 REFUSED，推进 |
| WebRTC 断线 | LiveKit 事件 | 触发 Redis 保存，等待重连 |
| GPU OOM | torch 异常 | 降级到 CPU 推理（慢但不崩） |

### 验收标准
- [ ] 故意输入噪音（低置信度），系统要求重复而非崩溃
- [ ] 故意不说话 10 秒，系统主动询问
- [ ] 故意说「这个我不方便说」，系统正确标记并推进
- [ ] 杀掉 LLM 服务，系统降级后继续运行（使用顺序题目）

---

## Phase 9 — 集成测试 & 延迟调优

### 目标
端到端延迟 P95 < 600ms；压测 5 并发会话无崩溃。

### 测试计划

```
测试类型        工具                  覆盖范围
──────────────────────────────────────────────────────────────
单元测试        pytest               状态机、触发评估、评分规则
集成测试        pytest + httpx       API 端点 + LLM mock
语音 E2E 测试   预录音频文件注入       完整访谈流程（模拟）
延迟测试        自定义 timing 日志    每个阶段耗时 P50/P95
压力测试        locust               5 并发会话，30 分钟运行
```

### 延迟测量点

```python
# 在 decision_loop 中打时间戳
async def decision_loop(...):
    while True:
        t0 = time.perf_counter()
        user_text, emotion = await asr_queue.get()
        t_asr = time.perf_counter()

        decision = await llm_judge(state, user_text, emotion)
        t_llm = time.perf_counter()

        audio = question_bank.get_audio(decision.next_question_id)
        t_audio = time.perf_counter()

        await audio_buffer.put(audio)
        t_put = time.perf_counter()

        logger.info("LATENCY asr=%.3f llm=%.3f audio=%.3f total=%.3f",
                    t_asr-t0, t_llm-t_asr, t_audio-t_llm, t_put-t0)
```

### 验收标准
- [ ] 用户感知响应延迟（音频输出开始）P95 < 600ms
- [ ] Barge-in 延迟 P95 < 80ms
- [ ] 5 并发会话下无内存泄漏（运行 30 分钟）
- [ ] 完整45分钟访谈端到端测试通过（使用预录音频）

---

## Phase 10 — 部署

### 目标
生产环境可访问，HTTPS，GPU 服务稳定运行。

### docker-compose.yml

```yaml
version: "3.9"
services:
  app:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [postgres, redis, livekit]
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  livekit:
    image: livekit/livekit-server:latest
    ports: ["7880:7880", "7881:7881"]
    command: --config /etc/livekit.yaml
    volumes: ["./livekit.yaml:/etc/livekit.yaml"]

  funasr:
    image: registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:latest
    ports: ["10095:10095"]
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: wealth_interview
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes: ["pgdata:/var/lib/postgresql/data"]

  redis:
    image: redis:7-alpine
    volumes: ["redisdata:/data"]

volumes:
  pgdata:
  redisdata:
```

### 验收标准
- [ ] HTTPS 访问正常（Let's Encrypt 或自签名）
- [ ] LiveKit TURN 服务配置（穿透 NAT）
- [ ] 服务重启后自动恢复（systemd / docker restart policy）
- [ ] 日志聚合可查（结构化 JSON 日志）
- [ ] 预生成音频脚本在部署时自动运行

---

## 关键里程碑

```
Week 1（Phase 0-2）：语音管道可通，题目音频可播放
Week 2（Phase 3-4）：完整对话可运行，分层触发工作
Week 3（Phase 5-6）：报告可生成，前端可用
Week 4（Phase 7-9）：持久化、异常处理、压测通过
Week 5（Phase 10）：生产部署，上线
```

---

## 待定事项（需业务侧确认）

| 编号 | 问题 | 影响 |
|------|------|------|
| T-1 | 核心层 30 个单元的具体选择 | 影响 Phase 2 题库建设 |
| T-2 | 260 个单元的完整字段定义表 | 影响 Phase 4 提取配置 |
| T-3 | 各单元评分规则（权重、阈值） | 影响 Phase 5 诊断引擎 |
| T-4 | 报告模板设计稿 | 影响 Phase 5 报告生成 |
| T-5 | 时间超限收尾策略（45分钟到了） | 影响 Phase 3 状态机 |
| T-6 | 是否需要顾问人工复核环节 | 影响整体流程设计 |

---

*开发计划 v2.0 | 基于架构设计文档 v1.0 | 2026-03-02*
