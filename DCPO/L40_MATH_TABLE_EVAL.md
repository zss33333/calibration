# 在 L40 上评测 Hopper 训练的 DCPO 模型

目标是论文 Table 1 的 **DCPO / verbal** 行：MATH-500、AIME24、AIME25、AMC23、AMC24，各独立采样四轮。每轮保存逐题 JSONL，并计算 Acc、ECE、PCE、AUROC。这里所有推理均在 L40 服务器进行；Hopper 只用于读取训练检查点。

## 1. 从 Hopper 复制第 120 步的 actor 分片

在 Hopper 登录节点找到实际实验名，并确认全部模型分片存在：

```bash
cd /scratch/weiliu87/student/zss/calibration/DCPO
find checkpoints/MATH -path '*/global_step_120/actor' -type d
ls -lh checkpoints/MATH/你的实验名/global_step_120/actor/model_world_size_*_rank_*.pt
```

在 **L40 服务器**运行下面的复制命令。设置你实际的 Hopper 登录地址和实验名；保留 `global_step_120/actor` 目录结构。只复制 `model_world_size_*_rank_*.pt` 即可用于合并，不需要 `data.pt`、optimizer 或 extra state。`rsync` 可在传输中断后续传。

```bash
HOPPER_LOGIN='weiliu87@你的Hopper登录地址'
RUN_NAME='你的实验名'
L40_ROOT=/home/weiliu/student/zss/calibration/DCPO
HOPPER_ACTOR=/scratch/weiliu87/student/zss/calibration/DCPO/checkpoints/MATH/$RUN_NAME/global_step_120/actor
L40_ACTOR=$L40_ROOT/checkpoints/MATH/$RUN_NAME/global_step_120/actor
mkdir -p "$L40_ACTOR"
rsync -avP --include='model_world_size_*_rank_*.pt' --exclude='*' \
  "$HOPPER_LOGIN:$HOPPER_ACTOR/" "$L40_ACTOR/"
```

检查分片数量及总大小，并确认 L40 上已经有基础模型与五个评测数据文件：

```bash
ls -lh "$L40_ACTOR"/model_world_size_*_rank_*.pt
test -f "$L40_ROOT/models/Qwen3-8B/config.json"
ls -lh "$L40_ROOT"/data/MATH-500/test.jsonl \
  "$L40_ROOT"/data/aime24.parquet \
  "$L40_ROOT"/data/aime25/aime2025.jsonl \
  "$L40_ROOT"/data/AMC/data/amc23-00000-of-00001.parquet \
  "$L40_ROOT"/data/AMC/data/amc24-00000-of-00001.parquet
```

若数据或基础模型缺失，先从已有存储同步到 L40。评测脚本要求五份数据至少分别有 500、30、30、45、45 行；它采用各文件的前这些行，与作者原评测脚本一致。

## 2. 在 L40 合并、部署并评测

先把本工作区的 `examples/l40_math_table_eval.sh`、`scripts/eval_math_table.py`、`scripts/model_merger.py` 同步到 L40 的 DCPO 目录。以下命令全部在 L40 运行，使用 L40 的 `.venv`，不用 Hopper 的 `.venv-hopper` 或 Singularity 镜像。确认有可用 L40 GPU 和足够的 CPU 内存、磁盘空间；合并时会在 CPU 内存中载入所有 actor 分片。

```bash
cd /home/weiliu/student/zss/calibration/DCPO
RUN_NAME='你的实验名'
export DCPO_PYTHON="$PWD/.venv/bin/python"
export DCPO_ACTOR_DIR="$PWD/checkpoints/MATH/$RUN_NAME/global_step_120/actor"
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
bash examples/l40_math_table_eval.sh
```

入口脚本先检查依赖与数据，然后运行 `model_merger.py --backend fsdp`，在实验目录的 `evaluation/step_120/hf_model/` 生成合并权重；接着用一张 L40 启动 vLLM，模型服务名为 `dcpo-step120`，并完成五个数据集的四轮评测。`vllm.log` 留在 `evaluation/step_120/`。若端口 8000 已占用，可设置 `DCPO_EVAL_PORT=8001`。如果所选 GPU 无法容纳当前 8192 token 的 vLLM 配置，应根据实际显存调整脚本中的 `--gpu-memory-utilization`、`--max-model-len`、`--max-num-seqs`，并记录调整内容。

L40 环境需要 `pandas`、`pyarrow`、`requests`、`scikit-learn`、`opencompass`、`vllm`。入口脚本会在启动服务前检查主要导入，缺依赖时直接报错。

## 3. 结果与无 GPU 重算

输出目录：

```text
checkpoints/MATH/你的实验名/evaluation/step_120/math_table_paper/
```

`MATH-500_run1.jsonl` 等文件每行是一道题，含题目、标准答案、完整模型输出、抽取后的答案、置信度与正确性。共应有 **20 个 JSONL**。`per_run.csv` 保存每轮的四个指标；`summary.csv` / `summary.json` 保存各数据集四轮指标的平均值。Acc 为 0–1 比例，填论文表时乘 100。ECE、PCE 使用 20 个等宽桶。入口脚本采用作者原评测代码的缺失置信度处理方式，并在汇总中明确报告 `invalid_confidence` 数量。中断后重跑入口脚本会跳过已有逐题记录。

模型服务停止后，无需 GPU 即可重新计算指标：

```bash
"$DCPO_PYTHON" scripts/eval_math_table.py \
  --data-root data \
  --output-dir "$(dirname "$(dirname "$DCPO_ACTOR_DIR")")/evaluation/step_120/math_table_paper" \
  --model dcpo-step120 --prompt-format paper --summary-only
```

默认 `paper` 使用论文附录的 `<conf> Confidence: 0.83` 提示。如果模型只稳定输出训练数据中的 `CONFIDENCE: 0.83`，可用 `DCPO_PROMPT_FORMAT=trained bash examples/l40_math_table_eval.sh` 另跑对照；它会写到独立目录，不能当作论文原提示口径。论文的 `Overall Performance` 聚合细节未明确，且论文 Overall Acc 与表中五项数值的简单平均不同，因此本脚本只输出五个可复核的数据集指标。
