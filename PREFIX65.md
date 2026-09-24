# 前65帧动力学训练

统一训练入口为 `bash scripts/train_prefix65_4gpu.sh train`，恢复使用 `resume`。
先设置 `DATA`、`MANIFEST`、`PREPARED`、`RESULT_ROOT` 和 `CUDA_VISIBLE_DEVICES`。
该入口使用本页既有四卡配置；RESULT_ROOT 使用新的独立目录。

本分支固定25个epoch，使用每条Train轨迹的stored frames 0–64，监督64个相邻帧转移。
原75帧数据文件和Train归一化继续复用。变化限定于动力学监督时间范围；归一化仍来自Train75帧。
评价保持首帧预测未来64帧、既有Validation选优与完整Validation100。Test保持封存。

四卡每卡batch1，全局batch4，每轮64,000个样本、16,000次更新；完整训练为1,600,000个样本、
400,000次更新。模型、噪声、优化器和学习率衰减规则沿用原配方。学习率按累计样本数递减，
因此25轮结束时的学习率随实际样本数确定。新增配置身份进入恢复记录，原75帧checkpoint保留。

## 论文结果回传

四组材料由现有评价、测速和训练曲线入口提供。所选权重为本次训练的304k checkpoint。
已有对应输出时直接收集文件；需要补跑时，在现有单H20推理环境执行下面两条命令。
`RUN`指向含`best.pt`、`epoch_*.pt`和`summary.json`的训练目录，通常为`RESULT_ROOT/main`。
`DATA`、`MANIFEST`、`PREPARED`沿用原数据路径，`OUT`设为新的结果目录。

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
RUN=/absolute/path/to/this_run/main
CHECKPOINT="$RUN/best.pt"
OUT=/absolute/path/to/mgn_paper_results
CAMPAIGN=cylinderflow_same_h20_20260920

python pareto_run.py \
  --config cylinderflow_config_prefix65_4gpu.json --checkpoint "$CHECKPOINT" \
  --dataset "$DATA" --manifest "$MANIFEST" --prepared "$PREPARED" \
  --campaign-id "$CAMPAIGN" --device cuda:0 --threads 2 \
  --output-dir "$OUT/selected"

python training_curve.py --run-dir "$RUN" \
  --config cylinderflow_config_prefix65_4gpu.json \
  --dataset "$DATA" --manifest "$MANIFEST" --prepared "$PREPARED" \
  --device cuda:0 --output-dir "$OUT/history"
```

第一条命令使用同一权重完成单H20测速和Validation100评价。测速沿用每条轨迹两次预热、
三次正式重复。第二条评测该run已保存的各checkpoint，并从权重中读取累计样本数、累计耗时
及GPU数，输出完整精度的`history.csv`。原`status.json`、每点日志和图片继续输出。
首次使用带成本导出的版本需新的history目录；之后重复同一命令可继续收集。

回传以下文件即可，原训练权重和其余预测保留在运行机器：

| 内容 | 文件 |
| --- | --- |
| 完整逐轨迹指标与时间序列 | `selected/quality/evaluation.json`；`selected/quality/candidate_000/`下的`summary.json`、`trajectory_metrics.csv`、`case_metrics.jsonl`、`frame_metrics.csv`、`failures.json` |
| 三个案例的65帧UVP与网格 | `selected/quality/candidate_000/predictions/trajectory_1095_seed_0.npz`，以及同目录的`trajectory_1013_seed_0.npz`、`trajectory_1086_seed_0.npz` |
| 原始计时、显存与配置 | `selected/timing/`下的`summary.json`、`protocol.json`、`samples.jsonl`、`samples.csv`、`failures.json`，以及`selected/point.json`、`selected/exit.json` |
| 训练曲线与完整训练成本 | `history/history.csv`、`history/status.json`、`history/exit.json`；原`RUN`下的`summary.json`、`run_manifest.json`、`selector.json`、`checkpoint_inventory.json` |

上述NPZ已经包含节点顺序、网格、参考场、时间和权重记录。100条评价中的轨迹编号可用于
提取原24条子集。显存原始记录以bytes为单位，耗时以seconds为单位。
`history.csv`的GPU-hours对应各checkpoint的累计训练时间；完整训练成本取原`RUN/summary.json`。
回传时检查所选权重记录的更新数为304000、Validation轨迹数为100、失败数为0，
并保留脚本生成的退出与失败记录。

若已有该权重的完整Validation100目录，直接回传其中同名指标及三份NPZ。
缺少同H20测速时可单独使用现有测速入口：

```bash
python -m cylinderflow.benchmark \
  --config cylinderflow_config_prefix65_4gpu.json --checkpoint "$CHECKPOINT" \
  --dataset "$DATA" --manifest "$MANIFEST" --prepared "$PREPARED" \
  --device cuda:0 --threads 2 --output-dir "$OUT/timing"
```

此时回传该timing目录的原始记录及已有评价的`evaluation.json`，两者均记录checkpoint身份。

## 启动

在新分支的独立checkout内复用既有环境，指定原数据、manifest、prepared和四张获配GPU：

```bash
python -m cylinderflow.run_four_gpu \
  --config cylinderflow_config_prefix65_4gpu.json \
  --dataset "$DATA" --manifest "$MANIFEST" --prepared "$PREPARED" \
  --gpus "$GPU_IDS" --output-dir "$RESULT_ROOT"
```

`RESULT_ROOT`使用全新目录。恢复同一运行时添加`--resume`。
原入口负责训练与Validation24选优；运行日志记录实际更新数和全局样本数。
训练完成后，以选中权重执行完整Validation100：

```bash
python -m cylinderflow evaluate \
  --config cylinderflow_config_prefix65_4gpu.json \
  --dataset "$DATA" --manifest "$MANIFEST" --prepared "$PREPARED" \
  --checkpoint "$RESULT_ROOT/main/best.pt" --mode validation \
  --device cuda:0 --output-dir "$RESULT_ROOT/validation100"
```

回传选中checkpoint的完整Validation指标、训练曲线、运行配置和退出记录。

## 验证范围

交付前仅执行静态语法、引用与diff检查。正式四卡环境由接收端核实，GPU运行结果待回传。
接收端如需缩短启动验证，复用既有正式入口的`--preflight`，保留四卡、生产模型和数据。
