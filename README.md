# My LLM Engine

轻量级大语言模型推理框架，从零开始构建的 LLM 推理引擎。

## 特性

- 支持 decoder-only Transformer 模型（LLaMA/Qwen 风格）
- KV Cache 优化，支持 Prefill + Decode 两阶段推理
- Paged KV Cache，支持动态内存分配
- 支持 GQA（Grouped Query Attention）
- 批量推理与请求调度（Continuous Batching）
- 支持从 HuggingFace 加载预训练模型（Qwen2等）
- RoPE 旋转位置编码
- 灵活的配置系统
- 轻量级性能分析工具

## 项目结构

```
model_inference/
├── my_llm_engine/                # 核心模块
│   ├── __init__.py               # 导出接口
│   ├── config.py                 # 配置类 (ModelConfig, EngineConfig, GenerationConfig)
│   ├── logging_utils.py          # 日志工具
│   ├── tensor_types.py           # 类型定义
│   │
│   ├── engine/                   # 推理引擎
│   │   ├── engine.py             # LLMEngine 主类
│   │   ├── kv_cache.py           # Dense KV Cache
│   │   ├── paged_kv_cache.py     # Paged KV Cache
│   │   ├── block_manager.py      # KV Block 内存池管理
│   │   ├── scheduler.py          # 请求调度器
│   │   ├── request.py            # 请求生命周期管理
│   │   ├── generation.py         # 生成函数
│   │   └── profiler.py           # 性能分析工具
│   │
│   └── models/                   # 模型实现
│       ├── transformer.py        # DecoderOnlyModel
│       ├── attention.py          # 注意力机制 (支持 MHA/GQA)
│       ├── layers.py             # 网络层 (RMSNorm, SwiGLU MLP)
│       ├── rope.py               # RoPE 位置编码
│       └── weight_loader.py      # 预训练权重加载
│
├── configs/                      # 配置文件
│   ├── tiny_model.json           # 小型测试模型配置
│   └── default_engine.json       # 默认引擎配置
│
├── examples/                     # 示例代码
│   ├── basic_generate.py         # 基础配置与日志示例
│   ├── tiny_generate.py          # KV Cache 推理示例
│   ├── multi_generate.py         # 批量推理示例
│   └── real_model_inference.py   # 真实模型推理示例
│
├── tests/                        # 测试用例 (64个测试)
│   ├── test_kv_cache.py          # KV Cache 测试
│   ├── test_paged_kv.py          # Paged KV Cache 测试
│   ├── test_engine_scheduler.py  # Engine/Scheduler 测试
│   └── test_tiny_model.py        # 模型组件测试
│
├── stage_*.txt                   # 阶段设计文档
└── pyproject.toml                # 项目配置
```

## 安装

```bash
# 克隆仓库
git clone https://github.com/159753a52/model_inference.git
cd model_inference

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -e .

# 安装真实模型推理所需依赖
pip install transformers safetensors sentencepiece accelerate

# 安装开发依赖（可选）
pip install -e ".[dev]"
```

## 快速开始

### 1. 基础使用 - 配置系统

```python
from my_llm_engine import ModelConfig, EngineConfig, GenerationConfig

# 创建配置
model_config = ModelConfig(
    vocab_size=32000,
    hidden_dim=256,
    num_layers=4,
    num_heads=4,
    num_kv_heads=4,  # GQA: 可以小于 num_heads
)

engine_config = EngineConfig(
    device="cuda",  # 或 "cpu"
    dtype="float16",
    max_seq_len=2048,
)

gen_config = GenerationConfig(
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

# 从 JSON 加载配置
model_config = ModelConfig.from_json("configs/tiny_model.json")
```

### 2. 使用随机权重模型测试

```python
import torch
from my_llm_engine import ModelConfig, EngineConfig, GenerationConfig
from my_llm_engine.models import build_decoder_only_model
from my_llm_engine.engine import generate

# 加载配置
model_config = ModelConfig.from_json("configs/tiny_model.json")
engine_config = EngineConfig(device="cpu", dtype="float32")

# 构建模型（随机初始化权重）
model = build_decoder_only_model(model_config, engine_config)

# 生成
prompt = torch.randint(0, model_config.vocab_size, (1, 10))
gen_config = GenerationConfig(max_new_tokens=20, do_sample=False)

output = generate(
    model=model,
    input_ids=prompt,
    model_config=model_config,
    engine_config=engine_config,
    gen_config=gen_config,
    use_kv_cache=True,
)
print(f"Generated: {output.tolist()}")
```

### 3. 加载真实模型推理（推荐）

```python
import torch
from my_llm_engine import GenerationConfig, EngineConfig
from my_llm_engine.models.weight_loader import load_model_from_modelscope
from my_llm_engine.engine import generate

# 配置 (GPU推理)
engine_config = EngineConfig(
    device="cuda",
    dtype="float16",
    max_seq_len=512,
)

# 加载预训练模型 (自动从 HuggingFace 镜像下载)
model, tokenizer, model_config, engine_config = load_model_from_modelscope(
    model_id="Qwen/Qwen2-0.5B",  # 支持 Qwen2-0.5B, Qwen2-1.5B, Qwen2-7B 等
    engine_config=engine_config,
)

# 生成
prompt = "Hello, my name is"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

gen_config = GenerationConfig(
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

with torch.no_grad():
    output_ids = generate(
        model=model,
        input_ids=input_ids,
        model_config=model_config,
        engine_config=engine_config,
        gen_config=gen_config,
        use_kv_cache=True,
    )

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)
# 输出示例: Hello, my name is Sarah and I am a 20 year old student...
```

### 4. 使用 LLMEngine 进行批量推理

```python
from my_llm_engine.engine import LLMEngine

# 创建引擎
engine = LLMEngine(
    model=model,
    model_config=model_config,
    engine_config=engine_config,
)

# 添加多个请求
prompts = [
    tokenizer.encode("Hello", return_tensors="pt").squeeze(),
    tokenizer.encode("World", return_tensors="pt").squeeze(),
]

gen_config = GenerationConfig(max_new_tokens=20)

for prompt in prompts:
    engine.add_request(prompt, gen_config=gen_config)

# 运行直到完成
engine.run_until_complete()

# 获取结果
for req_id, request in engine.get_all_responses().items():
    output = request.get_full_sequence()
    print(f"Request {req_id}: {tokenizer.decode(output)}")
```

## 核心组件说明

### ModelConfig

模型架构配置：
- `vocab_size`: 词表大小
- `hidden_dim`: 隐藏层维度
- `num_layers`: Transformer 层数
- `num_heads`: 注意力头数
- `num_kv_heads`: KV 头数（用于 GQA，可小于 num_heads）
- `intermediate_dim`: FFN 中间层维度
- `max_position_embeddings`: 最大位置编码长度
- `rope_theta`: RoPE 基频（Qwen2 使用 1000000）
- `attention_bias`: 是否使用 attention bias（Qwen2 为 True）

### EngineConfig

推理引擎配置：
- `device`: 设备（cuda/cpu）
- `dtype`: 数据类型（float16/float32/bfloat16）
- `max_seq_len`: 最大序列长度
- `max_batch_size`: 最大批次大小

### KVCache

KV 缓存用于优化自回归生成：
- **Prefill 阶段**：处理完整 prompt，填充 KV Cache
- **Decode 阶段**：每次只处理一个新 token，复用缓存
- **Paged KV Cache**：按 block 动态分配，支持更长序列

## 性能

在 NVIDIA A30 (24GB) 上测试 Qwen2-0.5B：
- 模型加载：约 10 秒
- 推理速度：约 25 tokens/s (float16)
- 显存占用：约 2GB

## 运行测试

```bash
# 运行所有测试 (64个测试)
pytest tests/ -v

# 运行特定测试
pytest tests/test_kv_cache.py -v

# 带覆盖率
pytest tests/ -v --cov=my_llm_engine
```

## 运行示例

```bash
# 基础配置测试
python examples/basic_generate.py

# KV Cache 推理测试
python examples/tiny_generate.py

# 真实模型推理（需要 GPU 和下载模型）
python examples/real_model_inference.py
```

## 支持的模型

当前支持加载以下模型架构：
- **Qwen2 系列**：Qwen2-0.5B, Qwen2-1.5B, Qwen2-7B 等（已验证）
- 其他 LLaMA 风格的 decoder-only 模型（理论支持）

## 开发阶段

项目分阶段开发，详见 `stage_*.txt` 文档：
- Stage 0: 项目骨架与配置系统
- Stage 1: MVP 前向推理
- Stage 2: KV Cache 优化
- Stage 3: Engine + Scheduler 多请求调度
- Stage 4: Paged KV Cache + 性能分析

## 依赖

- Python >= 3.8
- PyTorch >= 2.0.0
- transformers >= 4.40.0（用于加载预训练模型）
- safetensors（用于加载 safetensors 格式权重）
- sentencepiece（用于 tokenizer）

## License

MIT License
