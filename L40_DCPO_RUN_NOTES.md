# DCPO 在双 L40 服务器上的运行与排障记录

更新：2026-11-05。适用服务器的项目目录为 `/home/weiliu/student/zss/calibration/DCPO`，Python 为该目录下 `.venv/bin/python`，两张 GPU 为 NVIDIA L40。这份记录针对 **L40 软件链路检查**；Hopper H200 的 PBS 作业使用另一套环境和调度规则。

## 本次已确认的结论

- L40 环境输出：PyTorch `2.6.0+cu124`、CUDA `12.4`、vLLM `0.8.5.post1`、Ray `2.46.0`、Flash Attention `2.7.4.post1`。训练 parquet 有 10,000 行，验证 parquet 有 240 行。升级依赖后应重新核对。
- 该 L40 虚拟环境的 vLLM 已有 GPU UUID 支持；运行 `scripts/fix_vllm_uuid.py` 返回 `UUID support already present`。不要因为再次看到这行就反复修改 `site-packages`。先用实际的 Ray worker UUID 检查确认。
- 未禁用 P2P 时，两个 rank 均进入同一个 1 元素 `ALLREDUCE`，120 秒后一起 NCCL 超时；后续的 `SIGABRT`、Ray worker death 和 `GetTimeoutError` 是这次超时的后果，不能据此认定为显存不足。
- **本次在 L40 上有效的措施是 `NCCL_P2P_DISABLE=1`，并将其写入 L40 的 `.sh` 启动脚本。** 它禁用了 NCCL 的 GPU 间直接 P2P 传输。最新日志中 rank 0 和 rank 1 都打印了 `[model-load barrier] ... exit`，随后构建了 vLLM rollout。底层 P2P 故障的硬件或系统配置原因尚未确认；不要把这个设置直接复制到 Hopper H200 作业。
- 后续日志已出现 `step:2`、验证指标、`global_step_2` 的双 rank 保存记录，且检查脚本进入了续训进程，证明两步训练及验证流程已推进。粘贴内容在续训进程的数据过滤处结束，尚未见 `global_step_3` 或最终完成提示；不能据此声称 checkpoint 恢复已完成。
- 这两步的 `response_length/clip_ratio=1.000`、`confidence=-1`、`critic/advantages/mean=0`、`actor/pg_loss=0`、`actor/grad_norm=0`。这说明小型 L40 检查的响应全部到达 256 token 上限且该批次没有有效优势信号；它验证了训练流程能走通，但还不能验证正常的非零参数更新。H200 使用 3000 token 的原设置时必须重新看这些指标。

## 使用当前 L40 检查脚本

使用 `DCPO/examples/check_Qwen3_8B_L40_UUID.sh`。本仓库版本默认给这台 L40 检查设置 `NCCL_P2P_DISABLE=1`；若需对照实验，可显式传入 `NCCL_P2P_DISABLE=0`。它不会为 Hopper PBS 作业设置 `CUDA_VISIBLE_DEVICES`。

```bash
cd /home/weiliu/student/zss/calibration/DCPO
export DCPO_PYTHON="$PWD/.venv/bin/python"

# 先只检查 UUID 映射及两卡 NCCL all-reduce/barrier。
DCPO_NCCL_ONLY=1 bash examples/check_Qwen3_8B_L40_UUID.sh

# 通信通过后检查 8B 的两步训练、验证、保存和再续训一步。
bash examples/check_Qwen3_8B_L40_UUID.sh
```

更新后的检查脚本在上述两种模式中都会先用每次运行独立的 Triton 与 TorchInductor 缓存编译并运行一个小型 CUDA kernel。因此 `DCPO_NCCL_ONLY=1` 也可用于快速排查 L40 上的 `cuda_utils.so` 加载问题；通过结果不能代替 Hopper 镜像与 `.venv-hopper` 的验证。

不要仅凭 `Ray UUID workers` 判定 NCCL 已通过。成功的通信检查还应看到两个不同 GPU UUID、两个 rank 的 `all_reduce=3`，以及 `Ray NCCL all-reduce and barrier` 和 `NCCL-only preflight passed`。完整检查还应退出码为 0，并有 `global_step_2`、`global_step_3` 目录和脚本末尾的 `L40 software-path check completed`。日志和 checkpoint 保存在每次运行单独创建的 `checkpoints/MATH/Qwen3-8B_L40_UUID_CHECK_*` 目录。

## 再次卡住时按这个顺序查

| 现象 | 先检查 | 处理 |
| --- | --- | --- |
| `device_id_to_physical_device_id` 对 `GPU-...` 报 `ValueError` | 是否用了正确的 `.venv/bin/python`；Ray worker 是否各自只看到一个 UUID | 用当前环境的 Python 运行 `scripts/fix_vllm_uuid.py`，仅在脚本确认安装的 vLLM 属于受影响版本时打补丁；重新跑 UUID 检查。L40 上次检查已显示内建 UUID 支持。 |
| `ALLREDUCE` 或 barrier 卡住，之后 NCCL watchdog 超时 | 两个 rank 的 `CUDA_VISIBLE_DEVICES` 是否不同；启动日志是否显示 `NCCL_P2P_DISABLE: 1` | 先跑 `DCPO_NCCL_ONLY=1`。确认运行的是本仓库更新后的 L40 `.sh`；若仍失败，收集下方诊断输出。 |
| `[model-load barrier] ... enter` 后不再推进 | rank 0、rank 1 是否都进入，以及是否都退出 | `verl/workers/fsdp_workers.py` 的首个 barrier 显式使用 worker 的本地 CUDA device。单个 rank 未进入与两个 rank 均进入但无法退出是不同故障；先看 rank 日志。 |
| Ray 报 `AF_UNIX path length cannot exceed 107 bytes` | `RAY_TMPDIR`、`TMPDIR` 是否嵌在很深的项目或作业目录下 | 使用脚本创建的短路径 `/tmp/dcpo-ray.XXXXXX`；缓存与运行目录仍按运行实例分开。 |
| `source .../setup/validation_env.sh: No such file` | 是否误用了旧的 `Qwen3_L40_2GPU.sh` 或 `Qwen3-8B_L40_smoke.sh` | 当前工作区没有 `DCPO/setup/validation_env.sh`；优先用本文的独立检查脚本。 |

需要更详细的 NCCL 日志时，只在诊断运行中设置：

```bash
cd /home/weiliu/student/zss/calibration/DCPO
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,SHM,NET \
  DCPO_NCCL_ONLY=1 bash examples/check_Qwen3_8B_L40_UUID.sh
nvidia-smi topo -m
nvidia-smi topo -p2p p
```

如果禁用 P2P 后仍复现，应把上述日志和拓扑输出交给服务器管理员检查 P2P、PCIe、IOMMU/ACS 或容器配置；不要自行修改系统 BIOS 或 PCIe 设置。NVIDIA 文档说明这些配置可能导致 P2P 通信挂起。

参考：[NVIDIA NCCL 环境变量](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)、[NVIDIA GPU 通信排障](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)。

## 容易误读的日志与测试边界

- `enforce_eager=True` 下“启用 CUDA graph 才能使用 async output processing”是性能提示；不是当前报错。
- `LocalLogger is deprecated` 是弃用提示；`qwen3` 不在 MFU 统计支持列表会让 MFU 显示 0，不能用 MFU=0 判断训练未运行。
- 两张 L40 上的脚本使用缩短的响应长度、很小的 batch、BF16、CPU offload 等设置来检查软件流程。通过后仍不能证明 Hopper 上原论文配置的八卡显存、运行时长或 PBS/镜像环境已通过。
- Hopper 的 PBS 调度器管理 `CUDA_VISIBLE_DEVICES`；不要把 L40 脚本的手动 UUID 设备设置搬进 Hopper 作业。
