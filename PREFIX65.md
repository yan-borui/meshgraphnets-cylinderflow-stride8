# 前65帧动力学训练

本分支固定25个epoch，使用每条Train轨迹的stored frames 0–64，监督64个相邻帧转移。
原75帧数据文件和Train归一化继续复用。变化限定于动力学监督时间范围；归一化仍来自Train75帧。
评价保持首帧预测未来64帧、既有Validation选优与完整Validation100。Test保持封存。

四卡每卡batch1，全局batch4，每轮64,000个样本、16,000次更新；完整训练为1,600,000个样本、
400,000次更新。模型、噪声、优化器和学习率衰减规则沿用原配方。学习率按累计样本数递减，
因此25轮结束时的学习率随实际样本数确定。新增配置身份进入恢复记录，原75帧checkpoint保留。

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
