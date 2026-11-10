# Hopper H200 上运行 DCPO：环境、提交与排障记录

更新：2026-11-09。适用项目：`CFP03-SF-101`；Hopper 用户：`weiliu87`；DCPO 工作目录：`/scratch/weiliu87/student/zss/calibration/DCPO`。本文记录本项目已发生的故障和已验证的修复。没有完整日志时，将用户报告与日志验证区分记录。

## 先看当前验证状态

| 项目 | 状态与证据 | 仍不能推出什么 |
| --- | --- | --- |
| Hopper 一卡环境探针 | **已通过**。作业 `700799.hopper-m-02` 在 `hopper-16` 上退出码 `0`，输出 `Ray preflight passed`、`Compute preflight passed`、`HOPPER_PREFLIGHT_ONLY_PASSED`。 | 没有加载 8B 模型，也没有运行 DCPO 训练或多卡 NCCL。 |
| vLLM GPU UUID | **已在一卡 Ray worker 上动态验证**：该 worker 的 `CUDA_VISIBLE_DEVICES`、`ray_gpu_ids` 均为同一个 `GPU-...` UUID，vLLM 返回 `vllm_physical_id: 0`。 | 不能用登录节点的 `UUID support already present` 单独替代这个动态检查；也不能推断所有节点上的多卡流程已通过。 |
| Triton 和 `cuda_utils.so` | **已在一卡 H200 上验证**：作业独立缓存中生成 `cuda_utils.so`，探针用 `ctypes.CDLL` 加载，并成功编译、执行小型 Triton CUDA 内核。 | 不能推出训练中的每一种 Triton/Flash Attention 内核都通过。 |
| Hopper 一卡模型训练探针 | `examples/hopper_one_gpu_probe.pbs` 已有两步诊断训练与恢复分支；**尚未收到其完整通过日志**。 | 不能声称模型加载、反向传播、验证、保存或恢复已在 Hopper 通过。 |
| Hopper 四卡原负载五步测试 | 用户于 2026-11-09 报告五步测试已通过；尚未收到该成功作业的完整日志。早期 `700825`、`705045`、`705781`、`706275` 均为失败记录。 | 不能据此推算完整 5 epochs 的用时，也未验证正式训练的续跑。 |
| Hopper 八卡完整训练 | **未收到成功完成 5 epochs 的日志**。 | 一卡环境探针和 L40 测试均不能代替八卡正式运行。 |

双 L40 服务器的结论另见 [L40_DCPO_RUN_NOTES.md](L40_DCPO_RUN_NOTES.md)。那台服务器使用 `/home/weiliu/student/zss/calibration/DCPO/.venv`，不是 Hopper 的 `.venv-hopper`。

## 来源与适用规则

- 用户提供的《Hopper Cluster User Guide - CFP01 (version 1.1)》日期为 **2025-01-16**，示例项目是 CFP01，队列示例是 `auto`。其长期适用的原则包括：通过 PBS 提交 GPU 作业；容器使用 Singularity/Apptainer；用 `hpc project`、`hpc space`、`qstat`、`gstat` 检查项目、空间和作业；把 Hugging Face 与 Triton 缓存放到 `/scratch`；PBS 预留 GPU 后应用本身仍必须使用相同卡数。手册列出的容量、并发和队列限额可能已变化，提交前以登录节点当前命令为准。
- 本项目在 **2026-11** 的管理员邮件确认 `CFP03-SF-101` 自费项目可使用标准队列和高优先级队列。登录节点的 `hpc autox` 实际示例使用 `#PBS -q autox`。当前本项目的快速通道 PBS 文件因此使用 `autox`；不要把旧手册的 CFP01 项目号或 `auto` 机械复制进来。高优先级队列仍有各自并发限制。
- `hpc autox` 示例说明 CPU/内存按每卡 **12 CPU / 225 GB** 自动分配，支持单节点 1、2、4、8 GPU。实际 `qstat -fx 698117.hopper-m-02` 显示四卡作业进入 `mediumx`，资源为 48 CPU、900 GB；八卡作业日志曾显示 96 CPU、1800 GB。资源和队列策略应以每次 `qstat -fx` 为准。
- 用户提供的 `test_math.txt` 是另一项训练任务的 PBS **示例**，含作业数组、另一套虚拟环境和 `/scratch/weiliu87/cache` 路径；这些不是 DCPO 的运行要求。其 `rm -rf` 清理段也不要直接照搬。

## 固定路径和软件环境

| 用途 | 当前路径或设置 |
| --- | --- |
| 项目根目录 | `/scratch/weiliu87/student/zss` |
| DCPO 代码 | `/scratch/weiliu87/student/zss/calibration/DCPO` |
| Hopper 虚拟环境 | `DCPO/.venv-hopper`，进入容器后 `source .venv-hopper/bin/activate` |
| 训练模型 | `DCPO/models/Qwen3-8B`，应有 `config.json` 和 safetensors 分片 |
| 训练数据 | `DCPO/data/deepscaler_uniform_train_with_confidence.parquet`，历史检查为 10,000 行 |
| 验证数据 | `DCPO/data/test_data/aime24_repeated_8_times.parquet`，历史检查为 240 行 |
| 使用过且一卡探针通过的镜像 | `/app1/common/singularity-img/hopper/pytorch/pytorch_2.3.0_cuda_12.4_ngc_24.04.sif` |
| 正式/测试检查点 | `DCPO/checkpoints/MATH/...`；测试必须用独立的实验名和目录 |

镜像名称中的 PyTorch 版本**不是**激活 `.venv-hopper` 后实际导入的版本。作业 `700799` 输出：`torch 2.6.0+cu124`、CUDA `12.4`、`triton 3.2.0`、`vllm 0.8.5.post1`、`ray 2.46.0`、`flash-attn 2.7.4.post1`、`transformers 4.52.4`。更新环境或镜像后必须重测。模型权重和虚拟环境属于项目目录；Singularity `.sif` 是集群提供的公共镜像，不在 DCPO 项目目录下。不要凭镜像文件名推断依赖兼容性。

Hopper 登录节点用于准备、检查和 `qsub`；GPU 代码必须进入 PBS 分配的计算节点。登录节点上直接执行 `python scripts/preflight_hopper.py login` 曾调用 `/usr/bin/python`，因未进入虚拟环境报错。正确的无 GPU 检查方式是：

```bash
cd /scratch/weiliu87/student/zss/calibration/DCPO
module load singularity
IMAGE=/app1/common/singularity-img/hopper/pytorch/pytorch_2.3.0_cuda_12.4_ngc_24.04.sif
singularity exec -B /scratch:/scratch "$IMAGE" bash -c '
  cd /scratch/weiliu87/student/zss/calibration/DCPO &&
  source .venv-hopper/bin/activate &&
  python scripts/preflight_hopper.py login
'
```

登录节点若显示 `Could not find any nv files on this host`，不能据此判断计算节点 GPU 有故障。计算作业使用 `module load singularity`，再用 `singularity exec --nv -B /scratch:/scratch "$IMAGE" ...`。容器内激活项目虚拟环境；打印 `command -v python`、`torch.__version__`、`torch.version.cuda` 和 `nvidia-smi -L` 来确认实际执行环境。

## 提交与监控：可直接复用的步骤

登录节点先查项目、配额、当前快速通道示例和脚本。将本地工作区的新脚本**同步到 Hopper** 后再提交；PBS 传送作业脚本，不会自动传送脚本引用的 `scripts/*.py` 或模型文件。

```bash
cd /scratch/weiliu87/student/zss/calibration/DCPO
hpc project
hpc space
hpc autox
bash -n examples/hopper_four_gpu_full_train.pbs
bash -n examples/Qwen3-8B.sh
stat -c '%n | %s bytes | modified %y' \
  examples/hopper_four_gpu_full_train.pbs \
  examples/Qwen3-8B.sh scripts/preflight_hopper.py
grep -F 'DCPO_RAY_TEMP_DIR' verl/trainer/main_ppo.py
```

一卡**环境检查**不加载模型、不训练；此前 `700799` 已通过。若依赖、镜像、补丁或节点环境变化，可重跑：

```bash
qsub -v DCPO_PREFLIGHT_ONLY=1 examples/hopper_one_gpu_probe.pbs
```

四卡**完整训练**（原始 batch、rollout、验证频率和 5 epochs）：

```bash
qsub examples/hopper_four_gpu_full_train.pbs
```

这份四卡脚本沿用 `examples/Qwen3-8B.sh` 中的训练 batch 256、每题生成 8 条、提示上限 1024、回答上限 3000、actor mini batch 64、每卡 micro batch 8、vLLM `gpu_memory_utilization=0.6`、完整验证集和原来的奖励算法。现在不再覆盖 `trainer.total_training_steps`，所以按原脚本运行 5 epochs、每 20 步保存、每 5 步验证。保留四卡、Ray CPU 48、短 Ray 路径、作业独立缓存和 `actor_rollout_ref.actor.use_torch_compile=False`。`#PBS -l walltime=48:00:00` 是单次作业最长运行时间；若到时退出，固定的正式训练目录和 `resume_mode=auto` 允许用同一脚本续跑，但不要同时提交两个写入同一目录的作业。

提交后用返回的完整作业号：

```bash
qstat -awn1
qstat -fx <完整作业号>
qgpu_smi <完整作业号>   # 仅对正在运行的作业查看 GPU/CPU 负载
gstat
```

`Q` 表示排队；`qstat -fx` 的 `comment = Not Running: Insufficient amount of resource: ngpus` 表示调度器尚未找到满足资源请求的可用 GPU，**不是训练代码报错**。`gstat` 中节点 `ngpus` 的 `f/t` 是空闲/总数的瞬时快照，不保证作业马上启动。误提交时才用 `qdel <完整作业号>` 撤回。`#PBS -j oe` 合并 stdout/stderr；测试脚本的 PBS 输出在提交目录的 `dcpo4x5.o<作业数字>` 一类文件中，训练脚本还会写独立运行目录的 `training_process.log`。以实际 `qstat -fx` 的 `Output_Path` 为准。

## 路径、缓存、Ray 和并发隔离

所有可写缓存应放在 `/scratch/weiliu87/student/zss` 下，并按**作业 ID**隔离：`HF_HOME`、`HF_DATASETS_CACHE`、`XDG_CACHE_HOME`、`TORCH_HOME`、`TRITON_CACHE_DIR`、`TORCHINDUCTOR_CACHE_DIR`。四卡五步脚本使用 `/scratch/weiliu87/student/zss/cache/dcpo_smoke4/$PBS_JOBID` 作为缓存根；每个作业拥有自己的 `RUN_NAME` 和检查点路径。并发作业不要写同一个 `training_process.log`、验证/奖励输出或 checkpoint。

作业 `706700` 在第 1 步 `old_log_prob` 的 FlashAttention Triton `cross_entropy_fwd_kernel` 中，读取刚编译的 `.cubin` 时得到 `FileNotFoundError`。作业级缓存隔离仍让四个 Ray worker 共享同一 Triton 目录，最可能触发多进程同键写入竞争。四卡 PBS 现给预检单独的 `triton-preflight`，并设置 `DCPO_TRITON_CACHE_ROOT`；`verl/single_controller/ray/base.py` 在每个 `WorkerDict` 初始化时按 PID 设置独立的 `TRITON_CACHE_DIR`。提交前必须同时同步这两个文件，并在新作业日志中确认四条 `Ray worker Triton cache:` 指向不同目录。此改动仍待 Hopper 五步实测验证。

Ray 的 Unix socket 有长度限制。作业 `698724` 曾使用很深的 `.../cache/dcpo/<job>/tmp/ray/ray/session_.../sockets/plasma_store`，报 `AF_UNIX path length cannot exceed 107 bytes`。后来改用短路径：

```bash
JOB_NUM="${PBS_JOBID%%.*}"   # 必须在 set -u 下使用前定义
export TMPDIR="/scratch/weiliu87/student/zss/t$JOB_NUM"
export RAY_TMPDIR="/scratch/weiliu87/student/zss/r$JOB_NUM"
export DCPO_RAY_TEMP_DIR="$RAY_TMPDIR"
```

项目的 `verl/trainer/main_ppo.py` 应有 `ray.init(_temp_dir=os.environ.get('DCPO_RAY_TEMP_DIR'), ...)`。不能只设置 `RAY_TMPDIR` 却忽略代码实际传给 `ray.init()` 的变量。作业 `699320` 在 `set -u` 下先引用未定义的 `JOB_NUM`，23 秒就退出；这是 shell 变量顺序错误，不是 GPU 故障。CUDA/Triton 缓存同样保持每作业独立，以减少旧 `.so` 或并发写入影响；但一卡小型 Triton 测试通过不等于全部训练内核通过。

手册称 `/scratch` 不备份，旧文件可能被清理；`/local_ssd/$USERID` 是计算节点本地临时空间，不适合作为唯一检查点副本。该版手册还将 `/Project_Storage/...` 列为**仅登录节点可访问**，因此不要未经验证就把它设为计算作业的输出目录。定期用 `hpc space` 和运行目录大小检查配额；长期保存的模型与结果不要只依赖临时缓存。不要在未经核对的路径上运行示例文件中的递归清理命令。

## 历史故障：看到什么、如何判断

| 记录 | 根因或结论 | 以后怎么查 |
| --- | --- | --- |
| `698117` 四卡作业在 `mediumx` 排队，`Insufficient ... ngpus` | 资源未就绪，脚本尚未运行。 | 看 `qstat -fx`、`gstat`，不要改训练参数来“修复”排队。 |
| `698724` Ray plasma socket 路径超过 107 bytes | Ray 临时目录太深。 | 同时检查 `RAY_TMPDIR`、`DCPO_RAY_TEMP_DIR` 和 `main_ppo.py` 的 `_temp_dir`。 |
| `699320` `/usr/bin/bash: JOB_NUM: unbound variable` | `set -u` 下变量先使用后赋值。 | 在传入 `PBS_JOBID` 后立即定义 `JOB_NUM="${JOBID%%.*}"`。 |
| 2026-11-04 的四卡日志（例：`700050`）中 `ValueError: invalid literal for int(): 'GPU-...'` | 旧 vLLM `device_id_to_physical_device_id()` 把 PBS/Ray 提供的 GPU UUID 当数字。`WorkerDict failed to import` 与 `register_center_actor` 断言是后续症状。 | 使用正确 `.venv-hopper` 的 Python 检查 `scripts/fix_vllm_uuid.py`；它仅匹配已知旧实现才修改并备份。更重要的是用计算节点 Ray worker 动态验证。`700799` 已在一卡通过。 |
| 同一旧四卡日志先报 `MetricsHead failed to start` | Ray 后来仍启动并运行到数据加载/worker 创建；该作业的致命错误是 UUID。 | 不要仅凭 dashboard 警告认定整项训练失败原因；若未来只剩 dashboard 故障，再看该作业 Ray session 的 `dashboard.log`/`.err`。一卡环境探针使用 `include_dashboard=False`，并未验证 dashboard 启动。 |
| `700525` 计算节点称 `preflight_hopper.py` 不存在 | 作业约 01:27 结束，文件 01:37 才上传；运行时确实还没有。 | 提交前 `stat` 检查文件及时间；本地 `E:\calibration` 的编辑不会自动出现在 Hopper。 |
| 登录节点 `Not in a virtual environment: /usr/bin/python` | 在容器/虚拟环境外直接运行 `preflight_hopper.py login`。 | 用上述 `singularity exec` 和 `source .venv-hopper/bin/activate` 后再检查。 |
| `700785` Triton `NameError('tl is not defined')` | 当时的探针把 `triton.language as tl` 只放在局部作用域，是**探针自身缺陷**。 | 更新 `scripts/preflight_hopper.py` 后重跑；`700799` 已成功生成并加载 `cuda_utils.so`。无需为此重装 CUDA。 |
| `700799` 中 `Triton module has been replaced with a placeholder` | 本次仍完成 Triton 编译与 CUDA 执行，属信息行。 | 看同一作业最终是否有 `Compute preflight passed` 和退出码 `0`。 |
| `700825` 四卡五步作业打印 `main_ppo.py does not pass the short Ray directory to ray.init`，3 秒退出 | 这是 PBS 脚本的源码同步检查主动退出；Hopper 上的 `verl/trainer/main_ppo.py` 未包含本地已有的 `DCPO_RAY_TEMP_DIR` 补丁。模型加载、四卡 NCCL 和训练均未开始。 | 在 Hopper 登录节点检查 `ray.init` 实际代码；将 `_temp_dir=os.environ.get('DCPO_RAY_TEMP_DIR')` 加到调用中，或同步含该改动的源码，再运行 `python3 -m py_compile verl/trainer/main_ppo.py`。确认作业继续进入计算探针和训练后再判断其余问题。 |
| `705045` 四卡五步作业在第 1 步 `update_actor` 失败 | 配置、数据、四卡模型与 vLLM rollout 均已初始化；Ray 反序列化 `BackendCompilerFailed` 异常时又报 `missing inner_exception`，因此提交日志没有编译器内部错误。源码默认对 actor 熵函数使用 `torch.compile`，它是此阶段最明确的候选来源，尚不能仅凭本日志证明具体编译失败原因。旧 worker `.err` 中另有 `Trying to free a pointer not allocated here`，栈在 `Py_FinalizeEx`、`MemPool::~MemPool`，发生于进程退出清理时，不能把它直接认作编译首因。 | 五步 PBS 保留默认编译，设 `RAY_DEDUP_LOGS=0`、`HYDRA_FULL_ERROR=1`；`fsdp_workers.py` 的 `update_actor` 捕获分支在 worker 内打印原始 traceback，再以可序列化的 `RuntimeError` 传回 Ray。旧作业日志缺少本地源码的 `[model-load barrier]` 标记，重跑前先确认新版 `fsdp_workers.py` 确已同步到 Hopper，再同步 PBS。不要把 FSDP 弃用警告或 MFU 为零当作首因。 |
| `705781` 仅见 Ray 启动后以 PBS `Exit_status=-29` 结束 | 已确认请求 `06:00:00`、实际 `06:01:33`。`raylet.out` 显示 Ray 自动探测 `CPU: 112, GPU: 4`，而 PBS 此作业实际只分配 48 CPU；`raylet.err` 在 04:31:27 连续报告大量 worker 进程仍活着但未及时注册，之后没有 DCPO 训练日志。10:30:04 raylet 先收到 PBS SIGTERM，随后 dashboard agent 退出，不能把后者当作启动失败首因。贴出的 PBS 文件也未设置 `actor_rollout_ref.actor.use_torch_compile=False`，故该作业并未测试关闭编译。 | 四卡 PBS 显式设 `DCPO_RAY_NUM_CPUS=48`，`main_ppo.py` 将其传给 `ray.init(num_cpus=...)` 并打印 Ray 资源，避免 Ray 将整台节点的 112 CPU 作为本作业可用。提交训练前增加一个最多等待 300 秒的 Ray worker ping；若 worker 仍不能注册，可在几分钟内报错，避免再次耗尽六小时。同步两个文件后确认打印 `CPU: 48` 与 `Ray startup worker PID`。PBS 新版还设 `PYTHONUNBUFFERED=1`。不要只加长 walltime。 |
| `706275` 四卡五步作业再次报 `BackendCompilerFailed` 反序列化失败 | 与 `705045` 不同，栈定位到 `ray_trainer.py` 的 `compute_log_prob`，发生在第 1 步 actor 更新之前。四个 rank 的模型加载 barrier 均退出，vLLM rollout 已初始化；Flash Attention dtype 提示出现于加载早期，后续加载继续，不能据此归因。`dp_actor.py` 中 `compute_log_prob` 调用 `_forward_micro_batch`，其中默认 `torch.compile` 的熵函数是最直接的编译入口，但内部异常仍未见。 | 先查 `/scratch/weiliu87/student/zss/r706275/session_*/logs/worker*.err` 中最早的编译 traceback，并确认 Hopper 版 `fsdp_workers.py` 含 `Actor log-prob computation failed` 捕获分支。仅有 `Actor update failed` 捕获不足以覆盖这次异常；不要重复提交同一代码。 |
| `706275` 四卡五步重试在 `old_log_prob` 失败 | 四 rank 模型加载和 `[model-load barrier] ... exit` 均通过，vLLM rollout 初始化也完成；这次 Ray 报错位于 `ray_trainer.py:1110` 的 `compute_log_prob`，发生在 `update_actor` 之前。Ray 再次无法反序列化 `BackendCompilerFailed`，原有只包住 `update_actor` 的捕获没有覆盖该路径。 | 将 `fsdp_workers.py` 中 actor 与 reference 的 `compute_log_prob` 调用都包上原始 traceback 打印，并转换为可序列化的 `RuntimeError`；保留默认 `torch.compile`，下一次可在 worker `.err` 中看到编译器内部异常，再按其具体原因修复。不要把 FSDP/Flash Attention 初始化提示当作本次首因。 |
| `703885` RLCR 四卡作业在 `vllm/platforms/cuda.py` 的 `int(physical_device_id)` 报 `ValueError: 'GPU-...'` | RLCR 使用独立的 `.venv-hopper-rlcr`，该作业实际加载的旧 vLLM 未处理 PBS GPU UUID；这不能证明 DCPO 的 `.venv-hopper` 补丁失效。后续 `ChildFailedError` 与进程组警告不是首因。 | 在与训练相同的 Singularity 镜像内分别检查两套虚拟环境。DCPO 的 `preflight_hopper.py compute --expected-gpus 4` 会在同一计算作业中调用 vLLM 映射四个 GPU；只有该检查和后续训练实际通过，才算四卡路径验证完成。 |
| 登录节点直接调用 RLCR 的 `.venv-hopper-rlcr/bin/python`，`sysconfig` 指向不存在的 `lib/python3.9/site-packages/vllm` | 该登录节点解析出的解释器是 Python 3.9，而 RLCR 失败作业使用镜像内 Python 3.10 和 `lib/python3.10/site-packages/vllm`。这次静态检查没有读到作业实际使用的文件，不能据此判断补丁状态。 | 在与失败作业相同的 Singularity 镜像内，不加 `--nv` 做无 GPU 源码检查；先打印 `sys.version`、`sys.executable`、`sysconfig.get_paths()['purelib']` 和目标文件是否存在，再读补丁标记。 |
| PBS 结尾 `No GPU-related information available for this job` | 此前一卡成功作业也有这行，属于资源报告缺少 GPU 细项。 | 看 PBS Exit Status、探针成功标志和 `qgpu_smi`，不要把这行当作错误。 |

当前四卡正式训练 PBS 保留已通过五步测试使用的 `actor_rollout_ref.actor.use_torch_compile=False`。该项目的 `dp_actor.py` 只用此开关选择 `entropy_from_logits` 的编译版或原函数；数据规模、rollout、熵数值计算及 loss 路径仍在。它可能降低性能并改变显存峰值，不能证明其他 Torch/Triton 编译路径均可用；正式作业仍需检查参数更新、验证和 checkpoint。

`UUID support already present` 只是对当前虚拟环境源码的检查。Hopper 作业 `700799` 的 `Ray GPU worker` 结果才证明在 `hopper-16` 上，UUID 经 Ray worker 和 vLLM 实际走通。升级/重建 `.venv-hopper` 后应重新验证，不能盲目反复修改 `site-packages`。**PBS 自行设置 GPU 可见性；Hopper 作业和 Python 代码不要手动设置 `CUDA_VISIBLE_DEVICES`。**

L40 双卡曾用 `NCCL_P2P_DISABLE=1` 解决本机的 NCCL `ALLREDUCE` 超时；这是 L40 的实测措施，**不要无证据地搬到 Hopper**。Hopper 一卡探针不产生多 rank NCCL，因此四卡五步测试仍须观察 model-load barrier、NCCL、vLLM 初始化、反向传播和保存。如果四卡出现通信超时，先收集各 rank 最早的 NCCL 错误和节点拓扑，再决定措施；Ray 的 worker death 或 watchdog `SIGABRT` 常是上游通信超时的后果。

## 五步结果如何验收，以及正式训练前的边界

四卡五步测试只有在 PBS `Exit Status: 0`、日志末尾 `HOPPER_4GPU_5STEP_PASSED`、`latest_checkpointed_iteration.txt` 为 `5`、第 5 步四个 rank 的 `model/optim/extra_state` 分片和 `data.pt` 均非空时，才算脚本完整通过。第 5 步还应有 `Final validation metrics`；查看 `actor/grad_norm`、`actor/pg_loss`、优势和 `response_length/clip_ratio`，确认不只是流程走通。若它们持续为零或全部响应截断，应单独分析训练信号，不能把“成功退出”误称为有效学习。

原始 `examples/Qwen3-8B.sh` 是 batch 256、每题 8 条、回答上限 3000、5 epochs、每 20 步保存、每 5 步验证；历史日志按 10,000 行数据和 batch 256 计算为每 epoch 39 个完整 dataloader batch、总计约 **195 步**。`trainer.total_training_steps=5` 只缩短迭代数，不能证明 195 步一定能在给定 walltime 内完成。估算正式 walltime 时应测五步中的稳定步时、首次编译/模型加载、全量验证和 checkpoint 时间，并留出余量。调整 walltime 不会让训练自动完成；到期作业会终止，只有已成功写出的 checkpoint 才可考虑恢复。

从四卡切到八卡时，PBS 的 `select=1:ngpus=8` 和应用的 `trainer.n_gpus_per_node=8` **必须一致**。不要把 `model_world_size_4_rank_*` 的四卡 FSDP 检查点直接交给八卡 `resume_mode=auto`；加载器按当前 world size 寻找对应分片。正式训练使用专属检查点目录；若要从头训练，用 `resume_mode=disable`，若要在同卡数、同配置和同目录续训，再用 `resume_mode=auto` 并先核对 `latest_checkpointed_iteration.txt`。五步测试检查点是 FSDP 分片训练状态，不是已合并的 Hugging Face 模型。

实际显存利用率用运行中的 `qgpu_smi <作业号>` 和训练日志 `perf/max_memory_allocated_gb`、`perf/max_memory_reserved_gb` 判断。vLLM 的 `gpu_memory_utilization=0.6` 是 vLLM 自身配置，不代表整个进程只使用或保证使用 60% 显存。四卡能否跑原始负载须看四卡日志；一卡探针和“单张 H200 约 140 GB”的标称容量都不能代替实测。

## 本地文件与远程同步清单

本仓库中的以下文件是 Hopper 运行链路的一部分。修改后通过已有 SSH/文件传输方式同步到远程 DCPO 的**相同相对路径**，再在登录节点 `stat`、`bash -n`：

- `DCPO/examples/hopper_one_gpu_probe.pbs`：一卡环境或诊断训练探针。
- `DCPO/examples/hopper_four_gpu_full_train.pbs`：从原四卡五步 PBS 改名并修改的完整训练脚本。
- `DCPO/examples/Qwen3-8B.sh`：原始 DCPO 训练参数，支持 `DCPO_EXP_NAME` 与命令行覆盖。
- `DCPO/scripts/preflight_hopper.py`：登录节点、计算节点与 Ray worker 检查。
- `DCPO/scripts/fix_vllm_uuid.py`：仅供受影响的旧 vLLM 环境使用的幂等修复脚本。
- `DCPO/verl/trainer/main_ppo.py`：确认 `ray.init(_temp_dir=...)` 使用短的 `DCPO_RAY_TEMP_DIR`。
- `DCPO/verl/single_controller/ray/base.py`：四卡 Ray worker 的 Triton 缓存按进程隔离。

不要把 L40 的 `.venv`、UUID 设备设置或 `NCCL_P2P_DISABLE=1` 复制到 Hopper PBS。需要提交或查看最新队列政策时，重新运行登录节点的 `hpc autox`、`hpc project`，并核对本项目实际作业输出。
