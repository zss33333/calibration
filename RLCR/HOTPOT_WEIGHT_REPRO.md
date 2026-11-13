# HotpotQA：官方 RLCR 训练与奖励权重实验

这条路径使用 `official_upstream/` 中的官方 `rl_runner.py`、`GRPO_Trainer.py`、奖励、参数和 HotpotQA 配置。源码从 [damanimehul/RLCR 的 `be2624b` 提交](https://github.com/damanimehul/RLCR/tree/be2624b1753d738b9552df4dd6d50ac33b3668ee)逐字节复制，完整提交号见 `official_upstream/UPSTREAM_SOURCE.txt`，文件校验值见 `official_upstream/SHA256SUMS`。原有 `RLCR/` 根目录下的文件含梯度诊断修改，不能作为官方复现基准。本路径不会调用这些修改后的 runner、trainer 或奖励函数。

先按 [README.md](README.md) 安装环境、固定 TRL 版本并配置 Accelerate / DeepSpeed。官方配置面向 4 张 GPU，保留原始模型、数据、batch size、步数、采样、奖励函数和其他训练参数。

Hopper 的 `.venv-hopper-rlcr` 已在官方镜像内核对：TRL checkout 为官方 README 要求的 `69ad852e5654a77f1695eb4c608906fe0c7e8624`；Accelerate 1.3.0、Datasets 4.0.0、DeepSpeed 0.17.0、Transformers 4.48.3、vLLM 0.7.3、Triton 3.1.0、FlashAttention 2.7.4.post1 与官方环境文件一致，PyTorch 为对应 CUDA 12.4 构建的 2.5.1。

Hopper 本地 Qwen 模型的 `downloaded_revision.txt` 是 `d149729398750b98c0af14eb82c78cfe92750796`，与 [Qwen/Qwen2.5-7B 当前仓库 SHA](https://huggingface.co/Qwen/Qwen2.5-7B) 一致。Hotpot 本地数据有 20,000 条训练、500 条测试样本，字段与[官方数据集](https://huggingface.co/datasets/mehuldamani/hotpot_qa)一致，但本地副本未保存 revision；目前没有逐字节核实数据文件。这里的“官方对齐”特指训练源码、奖励实现、配置与主要依赖版本；模型/数据改用本地副本，关闭 Hub 推送和在线上报，并加入 Hopper 环境适配。

```bash
cd RLCR
# 官方权重基线：format=0.5, accuracy=0.5, brier=0.5
bash run_hotpot_baseline.sh

# 全部五组权重；包含同一个 0.50 基线，无须重复运行上面的命令
bash run_hotpot_weight_sweep.sh
```

`lambda` 表示正确性奖励权重，Brier 权重为 `1 - lambda`。五组分别是 `0.00/1.00`、`0.25/0.75`、`0.50/0.50`、`0.75/0.25`、`1.00/0.00`（正确性/Brier）。`format=0.5` 和两项 `0.000001` 的置信度统计奖励保持官方值。实验之间只改变正确性和 Brier 的奖励权重；为隔离结果，生成的配置另设 `run_name` 和 `output_dir`，并关闭自动推送 Hub。其余训练设置从官方配置继承。

可先生成配置而不启动训练：

```bash
DRY_RUN=1 bash run_hotpot_weight_sweep.sh
```

只运行指定组：

```bash
RUN_LABELS="0.25 0.75" bash run_hotpot_weight_sweep.sh
```

可通过 `CUDA_VISIBLE_DEVICES`、`NUM_PROCESSES`、`ACCELERATE_CONFIG`、`RUN_ROOT` 指定设备和输出位置。默认配置写在 `data/hotpot-weight-sweep/configs/`，模型分别保存在 `data/hotpot-weight-sweep/lambda-*/`，日志在 `data/hotpot-weight-sweep/logs/`。已有 checkpoint 时，官方训练入口会从对应输出目录继续训练。

## Hopper H200 / PBS

Hopper 上使用 [local_configs/hopper-four-gpu-weight-train.pbs](local_configs/hopper-four-gpu-weight-train.pbs)，每个权重提交一个四卡作业。该脚本在计算节点进入 Singularity 镜像和 `.venv-hopper-rlcr`，校验官方源码哈希，沿用四卡 DeepSpeed 配置，使用本地模型与数据，不覆盖 PBS 的 `CUDA_VISIBLE_DEVICES`。训练前单 rank 预计算官方数据映射缓存，避免四个训练进程同时写同一份缓存。

先将 `RLCR/official_upstream/`、`hopper_official_runner.py`、PBS 脚本以及 `DCPO/scripts/fix_vllm_uuid.py` 同步到 `/scratch/weiliu87/student/zss/calibration/` 的相同相对路径。PBS 脚本通过一个仅设置每 rank 缓存与临时目录的薄包装器 `hopper_official_runner.py`，再 `exec` 官方 `official_upstream/rl_runner.py`；它不修改训练算法、奖励或梯度。然后在 Hopper 登录节点检查文件并提交：

```bash
cd /scratch/weiliu87/student/zss/calibration/RLCR
bash -n local_configs/hopper-four-gpu-weight-train.pbs
test -f ../DCPO/scripts/fix_vllm_uuid.py
test -f local_configs/hopper-four-gpu-full-step.yaml
test -f hopper_official_runner.py
cd official_upstream && sha256sum -c SHA256SUMS && cd ..
qsub -v RLCR_PREFLIGHT_ONLY=1 local_configs/hopper-four-gpu-weight-train.pbs
```

预检输出须包含 `RLCR_HOPPER_PREFLIGHT_PASSED`，并确认 PBS 退出码为 0。接着可用同一原始 batch 跑一次优化器更新，检查模型保存：

```bash
qsub -v RLCR_LAMBDA=0.50,RLCR_MAX_STEPS=1 local_configs/hopper-four-gpu-weight-train.pbs
```

看到 `RLCR_HOPPER_WEIGHT_TRAIN_PASSED` 且 PBS 退出码为 0 后，提交完整训练。例如：

```bash
qsub -v RLCR_LAMBDA=0.50 local_configs/hopper-four-gpu-weight-train.pbs
qsub -v RLCR_LAMBDA=0.25 local_configs/hopper-four-gpu-weight-train.pbs
```

其余权重同理。完整训练的输出按权重固定在 `data/hopper_weight_train/lambda-*/`，失败重提会从该目录中的官方 checkpoint 续跑；同一权重并发运行会被文件锁阻止。`RUN_COMPLETE` 和 `RLCR_HOPPER_WEIGHT_TRAIN_PASSED` 只会在训练结束、状态与模型权重检查通过后出现。PBS 脚本不会运行梯度冲突诊断入口。

### 保存与断点续跑

官方 Hotpot 配置是 `save_strategy=steps`、`save_steps=60`、`save_total_limit=1`，Hopper 正式训练原样保留。四卡生成 batch 为 `4 × 8 × 64 = 2048` 条生成结果，每题 `num_generations=32` 次，所以每批是 64 道不同题目。官方 `RepeatSampler` 丢弃不足 64 题的尾批；20,000 道训练题目形成 `floor(20000 / 64) = 312` 个完整生成批，每批经过 64 次梯度累积完成一个优化器更新。因此当前一轮预计 **312 个优化器步骤**，会在第 60、120、180、240、300 步保存。先前“约 10 步”的计算漏掉了每题 32 次生成，是错误的。

每个 `checkpoint-N/` 包含模型、DeepSpeed 优化器分片、调度器/RNG 状态和 `trainer_state.json`，与结束时单独保存的模型权重不同。Transformers 在优化器和 RNG 保存后才写 `trainer_state.json`；脚本重提时选择有此标记的最新 checkpoint，若最新目录在保存中途被作业终止则跳过。正常完成时只保留最近的 `checkpoint-300/`，最终模型在输出目录根部。脚本检查最近 checkpoint、四个非空优化器分片和最终模型权重；成功才写 `RUN_COMPLETE`。同一权重、同一四卡配置和输出目录可重新 `qsub` 续训；改变训练配置时配置比对会阻止误续跑。一优化器步骤探针不保存中途 checkpoint，只检验训练与最终模型保存。

DCPO 当前 Hopper 正式脚本是每 20 步保存、`resume_mode=auto`，但 `max_actor_ckpt_to_keep=null` 并不等于只保留一个历史 checkpoint。此处按“只留最近一个”的要求使用 RLCR/Transformers 的 `save_total_limit=1`。

当前仓库记录的 RLCR Hopper 故障与对应检查：

| 已记录问题 | 本脚本的检查或处理 |
|---|---|
| 作业 `703885`：旧 vLLM 将 PBS GPU UUID 传给 `int()`，出现 `ValueError` | 在训练容器和同一虚拟环境中、启动 Accelerate 之前，带锁运行已知版本的 UUID 修复脚本，并检查四个 GPU 的 vLLM 映射互不重复。 |
| 登录节点的 Python 3.9 路径与作业中的 Python 3.10 路径不一致 | 在计算容器里检查解释器版本和实际 `vllm/platforms/cuda.py` 文件，然后才修复和训练。 |
| 作业 `713616`：`reused_gradient_trainer.py` 抛出 `Training/sidecar full gradients disagree`，后续出现 `ChildFailedError` | 这是梯度冲突检测实验自身的校验失败；本路径只调用 `rl_runner.py`，不加载 `reused_gradient_trainer.py`、`pre_step_hook.py` 或诊断入口。 |
| 作业 `713786`：诊断入口的一步训练通过，输出 `RLCR_ONLINE_FULL_PASSED`，但按探针设置跳过最终模型保存 | 这证明该作业的四卡 UUID 映射、数据加载及一次优化器更新已走通；本路径取消诊断探针的跳过保存环境变量，并在训练后检查模型权重文件。 |
| 作业脚本或依赖文件尚未同步到 Hopper | PBS 开头检查镜像、虚拟环境、官方配置、DeepSpeed 配置及修复脚本；提交前仍需在登录节点同步并核对。 |
| 本地模型或 HotpotQA 数据缺失、结构不符 | 启动训练前加载本地模型配置和 tokenizer，实际加载数据并验证 `train`/`test`、`answer`、`source` 以及训练代码接受的 `question` 或 `problem`。Hopper 当前数据实际使用 `problem`。 |
| DCPO 早期四卡作业 `706700` 的 Triton 缓存 `.cubin` 文件竞争 | RLCR 的四个 Accelerate rank 使用各自独立的 Triton 缓存和临时目录；此措施不改变官方训练计算。 |
| DCPO 近期作业 `713604` 出现 `.__dpc... Device or resource busy` | 日志中该异常发生在 Python 临时目录清理，之后训练继续到第 23 步，不能当作该作业训练首因；RLCR rank 临时目录独立，降低互相清理同一目录的风险。 |
| DCPO 早期 `BackendCompilerFailed` 和短 Ray 目录问题 | RLCR 当前脚本没有显式启用 `torch.compile`，近期 RLCR 日志也显示 `torch_compile=False`；但 vLLM 等依赖仍可能触发 Ray 或编译。包装器为每个 rank 设置短 `RAY_TMPDIR` 和独立 `TORCHINDUCTOR_CACHE_DIR`，并关闭 Ray 日志去重，便于在新日志中定位首个异常。不能据此保证编译故障不会发生。 |
| L40 的 NCCL P2P 超时 | 这不是 Hopper 已确认的故障；Hopper 脚本不设置 `NCCL_P2P_DISABLE=1`，保留原始通信设置。 |

`713616` 和 `713786` 的原始 PBS 输出目前可在 Hopper 的 `RLCR/zssrlcr4t.o713616` 与 `RLCR/zssrlcr4t.o713786` 查到。`703885` 的完整 PBS 原始日志未找到，其首因依据现有故障记录。新脚本尚未在 Hopper 运行；最终还需以计算节点预检、单步训练和完整训练日志确认环境与运行结果。
