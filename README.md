# Calibration

数学推理与置信度校准实验工作区，包含 DCPO 和 RLCR 的代码及本地环境配置。

- `DCPO/`：Decoupled Calibration Policy Optimization，基于[作者仓库](https://github.com/icip-cas/DCPO)。本机环境使用方法见 [DCPO/setup/SETUP.md](DCPO/setup/SETUP.md)。
- `RLCR/`：Reinforcement Learning with Calibration Rewards，基于[作者仓库](https://github.com/damanimehul/RLCR)。包含本地保留的固定版本 TRL 依赖，使用方法见 [RLCR/README.md](RLCR/README.md)。

两个项目以普通源码目录保存，克隆本仓库即可取得代码。各项目原有的版权、许可证和引用要求保留在对应目录中。

下载的模型权重、训练/评估数据、评估输出、缓存和本机凭据不纳入此仓库。DCPO 使用的 Qwen3-8B 来自 Qwen 官方发布，数学数据来自作者仓库；本地环境与数据配置说明保留在 `DCPO/setup/`。
