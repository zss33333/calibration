# DCPO 数学主表：Hopper 评测步骤

本流程评测单个 DCPO 检查点，生成论文数学主表中 DCPO 的五个数据集结果。训练期间的 AIME24 验证不等同于该独立测评。默认使用论文附录所说的第 120 步、四次独立采样，以及论文的 `<conf> Confidence: ...` 提示格式。

## 1. 在登录节点检查文件与环境

先将 `scripts/eval_math_table.py`、更新后的 `scripts/model_merger.py` 和 `examples/hopper_math_table_eval.pbs` 同步到 Hopper 的 `/scratch/weiliu87/student/zss/calibration/DCPO`。PBS 提交时不会自动同步这些文件。

提交前在登录节点运行 `python scripts/eval_math_table.py --data-root data --check-data-only`（使用项目环境）。当前脚本会打印五个数据集的题数，最后打印 `EVAL_DATA_CHECK_PASSED`。若命令退出码为 0 却没有这些输出，说明服务器上的脚本不是当前版本；不要据此提交 GPU 作业。作业结束还会检查 `summary.csv`、`summary.json` 和 20 个逐题 JSONL，缺任何文件都会以非零状态退出。

若旧版合并脚本在 `from verl.utils.megatron_utils ...` 处报 `No module named 'megatron'`，同步本工作区的新版 `model_merger.py` 即可；FSDP 合并不依赖 Megatron。提交前可在 Hopper 登录节点运行 `grep -n megatron_utils scripts/model_merger.py`，确认导入只位于 `convert_megatron_checkpoints_to_hfmodes()` 函数内部。

```bash
cd /scratch/weiliu87/student/zss/calibration/DCPO
RUN_NAME=Qwen3-8B_HOPPER_4GPU_FULL
ls -lh "checkpoints/MATH/$RUN_NAME/global_step_120/actor"/model_world_size_*_rank_*.pt
ls -lh data/MATH-500/test.jsonl data/aime24.parquet \
  data/aime25/aime2025.jsonl \
  data/AMC/data/amc23-00000-of-00001.parquet \
  data/AMC/data/amc24-00000-of-00001.parquet
```

检查 `.venv-hopper` 中是否有 `pandas`、`requests`、`scikit-learn`、`opencompass`。`opencompass` 是正确率评分器，缺失时应先在项目环境中安装，不能省略评分步骤。脚本在 PBS 作业加载模型前会再次检查依赖和数据集行数。五个数据集按作者原评测脚本取前 500、30、30、45、45 行。

## 2. 提交 GPU 评测作业

```bash
qsub -v DCPO_RUN_NAME=Qwen3-8B_HOPPER_4GPU_FULL,DCPO_EVAL_STEP=120 \
  examples/hopper_math_table_eval.pbs
```

该作业申请 1 张 GPU、2 小时 walltime，先用 `model_merger.py` 将 `global_step_120/actor` 的 FSDP 模型分片合并到 `evaluation/step_120/hf_model/`，再以该目录启动 vLLM。服务名是 `dcpo-step120`。随后在 MATH-500、AIME24、AIME25、AMC23、AMC24 上分别独立采样四轮，共 2600 次生成。原始模型的 tokenizer 由 `models/Qwen3-8B` 提供。2 小时是否足够取决于实际输出长度和 GPU 吞吐；若作业因 walltime 结束，重新提交同一脚本会复用已完成的合并模型与逐题 JSONL，继续缺失题目。

若所评模型在论文格式下无法稳定输出 `<conf> Confidence: ...`，可以用与当前工作区训练数据一致的 `CONFIDENCE: ...` 格式另做对照：

```bash
qsub -v DCPO_RUN_NAME=Qwen3-8B_HOPPER_4GPU_FULL,DCPO_EVAL_STEP=120,DCPO_PROMPT_FORMAT=trained \
  examples/hopper_math_table_eval.pbs
```

两种提示格式的结果写入不同目录。`trained` 格式与论文附录的提示格式不同，报告时必须标明。

## 3. 检查结果和重新汇总

默认输出目录为：

```text
checkpoints/MATH/Qwen3-8B_HOPPER_4GPU_FULL/evaluation/step_120/math_table_paper/
```

每个 `数据集_run编号.jsonl` 文件包含每道题的题目、标准答案、完整模型输出、抽取的答案、置信度和正确性。已有记录在重新提交同一作业时会跳过。原始记录中的缺失置信度始终保留为 `null`，汇总同时报告 `invalid_confidence` 数量。Hopper 与 L40 入口脚本显式选用 `--invalid-policy author`：与作者发布的 `calibration_main.py` 一致，缺失 verbal confidence 按 `-0.1` 参与 AUROC，ECE/PCE 分桶时跳过它但仍以所有样本数为分母。若要强制所有样本有合法置信度，使用 `--invalid-policy strict`；它会在缺失时停止汇总。

若旧版 `eval_math_table.py` 报 `ValueError: signal only works in main thread of the main interpreter`，应同步本工作区新版脚本后重提作业。新版只在工作线程中发推理请求，在主线程导入 OpenCompass 并评分。已完整写入的逐题记录会自动跳过，空的 JSONL 会从头补齐；合并模型已带 `.merge_complete` 时不会重复合并。

每条请求还记录独立的采样 `seed`（默认基数 20260310），以便核对四轮采样；重新提交时需保持相同基数。

若已生成 JSONL 但旧版脚本因 `Missing/invalid confidence` 没有汇总，先不要删除记录。同步新版 `scripts/eval_math_table.py` 后，可在合适的 Python 环境中运行 `python scripts/eval_math_table.py --output-dir checkpoints/MATH/你的实验名/evaluation/step_160/math_table_paper --audit-confidence`。它只读取现有结果并写出 `confidence_audit.json`。新生成的记录还包含 vLLM 的 `finish_reason` 与生成 token 数，便于识别长度截断。

`per_run.csv` 是每轮指标，`summary.csv` 和 `summary.json` 是四轮指标的平均值。Accuracy 存为 0–1 比例，填论文表时乘 100；ECE、PCE、AUROC 保持 0–1。ECE 和 PCE 使用 20 个等宽置信度桶，与仓库原评测函数的默认值一致。若任一轮全部正确或全部错误，AUROC 无定义，结果会记录为空值。

完整 `summary.csv` 只在五个数据集各四轮全部结束后写出。若作业仍在运行或被时限截断，可在登录节点对已完整写入的轮次生成**部分结果**（无需 GPU）：

```bash
python scripts/eval_math_table.py --data-root data \
  --output-dir checkpoints/MATH/Qwen3-8B_HOPPER_4GPU_FULL/evaluation/step_160/math_table_paper \
  --model dcpo-step160 --prompt-format paper --summarize-existing
```

查看 `partial_per_run.csv` 和 `partial_summary.csv`；后者的 `runs_completed` 标明每个数据集当前完成的轮数，不能当成四轮最终结果。PBS 脚本默认 walltime 已设为 24 小时，重提同一步作业时会跳过已有逐题记录。

完整结果已生成后，可以在无需 GPU 服务的环境中重算指标：

```bash
python scripts/eval_math_table.py \
  --data-root data \
  --output-dir checkpoints/MATH/Qwen3-8B_HOPPER_4GPU_FULL/evaluation/step_120/math_table_paper \
  --model dcpo-step120 --prompt-format paper --summary-only
```

论文的 `Overall Performance` 聚合方式未在评测协议中明确，且表中 Overall Acc 与五列已发表数值的简单平均不一致。本脚本只输出五个可复核的数据集结果，不把自定义汇总数冒充论文的 Overall 数值。要复现完整方法对照表，还需分别取得 Base、GRPO、RLCR、CCGSPG、DCPO-G 和 DCPO-I 的模型或训练检查点，并按同一评测口径运行。
