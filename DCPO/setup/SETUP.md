# DCPO 环境使用说明

Conda 环境：`/home2/zhousishan/miniconda3/envs/DCPO`。原 `rlcr` 环境保持不变。

```bash
source /home2/zhousishan/miniconda3/etc/profile.d/conda.sh
conda activate DCPO
cd /home2/zhousishan/calibrate/DCPO
source setup/common.sh
```

核心版本为 Python 3.10、PyTorch 2.6.0（CUDA 12.4）、vLLM 0.8.5.post1、Transformers 4.52.4、FlashAttention 2.7.4.post1。完整 Python 版本清单保存在 `requirements.lock.txt`，Conda 包清单保存在 `conda-explicit.txt`。

## 模型与数据

- Qwen3-8B 完整权重与 tokenizer：`models/Qwen3-8B/`，来自 Qwen 官方 ModelScope 发布。
- 作者原始数学训练数据：`data/deepscaler_uniform_train.parquet` 和 `data/deepscaler_uniform_train_with_confidence.parquet`，各 10,000 题，未自行抽样或改写。
- 作者评估数据：`data/test_data/`，包括 MATH-500、AIME24/25 和 AMC23/24；Git LFS 文件已下载为实际内容。
- `data/MATH-500`、`data/AMC`、`data/aime25`、`data/aime24.parquet` 为兼容原脚本路径的链接。

数据来自作者仓库提交 `2ebb9770d33fc045fff45329a542a476cfcd8e13`。所有资源通过官方站点或镜像直连取得，下载完成时已校验哈希。环境、CUDA、FlashAttention 及 Transformers/vLLM 短生成均已通过检查；未运行训练或完整评估。

安装下载缓存、临时文件、验证脚本、验证报告、日志及一次性搭建脚本已清理。保留的 `common.sh` 在使用时按需创建工作区缓存目录，并仅对当前 shell 设置缓存路径和直连配置。

## 运行前仍需处理的代码问题

- 默认训练脚本选择 `hybrid`，但 `verl/trainer/main_ppo.py` 尚未提供该选择分支。
- 校准奖励的 rollout 分组数量仍写死为 8，完整训练行为尚未验证。
- `examples/eval.sh` 调用的 `Uncertainty/calibration_main.py` 不存在；实际脚本为仓库根目录的 `calibration_main.py`。
- AMC23 原数据为 46 题，评估加载器默认只取前 45 题。
- 默认训练配置面向 8 × A100 80GB；当前 8 × RTX 3090 24GB 尚未验证能够运行该配置。
