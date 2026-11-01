# Agent → Robot：个人探索与代码摘录

探索通用 Agent 如何通过明确的机器人控制接口，形成“观察—行动—检查结果—调整”的闭环。工作包括 Direct-EEF / Code-as-Policy 框架复现、原生评分评测、教师轨迹整理、小模型 LoRA 训练和工具调用失败分析。目前主要在仿真中开展。

## 代码入口

| 目录 | 内容 | 运行条件 |
| --- | --- | --- |
| `data/contract.py` | 单步动作协议、参数约束、reasoned / action-only 格式 | Python 3.10+ 标准库 |
| `data/export_episode.py` | 将已完成且通过原生成功校验的教师回合导出，绑定请求、回复、图像与执行反馈 | 原始回合目录、图像及质量记录 |
| `training/qwen_sft_sample.py` | 多模态样本编码与最终 assistant 答案监督 | 兼容模型的 Transformers、PyTorch |
| `training/train_lora.py` | Qwen3.5 2B / 4B LoRA 的样本预检、训练与记录 | 模型权重、数据、GPU、PEFT |
| `diagnostics/` | CaPtain 轨迹、工具调用链与结果口径分析 | Linux、jsonschema、兼容的原始评测日志 |
| `evidence/metrics.json` | 归档中的实验统计摘要 | 可直接阅读；不是本次重新评测 |

这些是可审阅的研究组件，**不是包含所有仿真环境的一键运行 SDK**。CaPtain、RoboDojo、LIBERO、SimplerEnv 等上游框架、数据、权重与基准资产需要分别准备。省略了上游整包、服务器配置、进程管理 launcher 和原始调用日志。

## 数据与训练

整理 12 条教师交互中的 894 个相关决策状态，按完整布局划分数据；同一状态的两种输出格式不计作独立样本。训练记录包括 Qwen3.5-2B / 4B 三轮 LoRA。少量闭环核对未证明成功率提升，验证 loss 下降不能代替机器人任务成功。

在准备好与归档格式一致的数据和模型之后：

```bash
export LORA_RUN_DIR=/path/to/your/run
export LORA_MODEL_DIR=/path/to/your/model
mkdir -p "$LORA_RUN_DIR"/{dataset,reports,encoded,logs,checkpoints}
python training/train_lora.py --prepare
python training/train_lora.py
```

请先查看脚本参数：准备阶段需要 `dataset/all.jsonl`、训练/验证划分及 `reports/dataset_snapshot.json` 中的源文件哈希。数据不随本仓库提供，空目录不足以运行。模型客户端版本未在归档中锁定，此次未重新训练。诊断脚本保留原分析逻辑，机器路径已经替换为 `/path/to/workspace`，使用前需按自己的日志位置调整。

## 实验记录的口径

- Direct-EEF：15 个有效短测场景中 2 个成功，不是官方全量基准。
- CaPtain-4B / LIBERO：400 轮，环境成功 300 轮；环境成功且主动 finish 为 234 轮。
- CaPtain-4B / SimplerEnv：1,400 轮，环境成功 292 轮；环境成功且主动 finish 为 228 轮。
- 不同模型、接口与任务集合不能当作受控算法排行榜。工具错误可能出现在最终成功的回合中，也不能直接等同于整局失败原因。

后续重点是动作粒度、观测与反馈一致性、未见布局泛化，以及通过公开图像和工具回执判断失效并恢复。未完成的扩展实验不计入最终成绩。

## 来源、完整性与许可

用户提供的 ZIP 在图表文件处截断，缺少中央目录。此次只从完整的 ZIP 本地文件条目中恢复并逐条验证 CRC，再筛选上述源码；**并未发布整个残缺归档或声称恢复所有内容**。`provenance.json` 记录所选条目及整理前后 SHA-256。

数据协议、导出、训练与诊断脚本来自个人实验归档；CaPtain 相关许可和第三方说明保留在 `licenses/`。上游算法和模型归原作者所有，个人贡献主要为复现、协议检查、数据整理、训练实验和评测诊断，不宣称重新提出这些框架。未对无统一许可证的个人脚本擅自追加开源授权。

发布检查包括源码语法、动作协议的合法/非法输入检查、文件完整性与敏感信息模式扫描；未重新运行 GPU 训练或完整仿真评测。
