# 前65帧动力学训练

Airfoil 的 MGN 使用 stored frames 0–64 训练，覆盖前 64 个时间间隔。
MGN 每条轨迹遍历 64 个相邻帧对，即 0→1 至 63→64。
原 75 帧 HDF5、Train75 归一化、模型、优化器、学习率规则和 Validation 选优规则沿用既有配方。

四卡配置每卡 batch1、accumulation1，全局 batch4。
25 轮；每轮 64,000 个样本、16,000 次更新；全程 1,600,000 个样本、400,000 次更新。
学习率继续按累计训练样本数衰减。
评价从 frame0 预测未来 64 帧。时间间隔为 0.0016，预测全节点 UVP。
Test 保持封存。

## 训练与恢复

复用已经安装的四卡训练环境，以及该数据集、该方法已有的数据准备目录。
首次准备数据可先运行 `bash scripts/airfoil_4gpu.sh prepare`，然后将 PREPARED 指向生成的 prepared_mgn_4gpu 目录。
路径使用绝对路径。RESULT_ROOT 使用新的独立目录。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA=/absolute/path/to/airfoil_stride8_75frames.h5
export MANIFEST=/absolute/path/to/matching_manifest.json
export PREPARED=/absolute/path/to/prepared_mgn
export RESULT_ROOT=/absolute/path/to/airfoil_mgn_prefix65
bash scripts/train_prefix65_4gpu.sh train
```

恢复同一次训练，保留上述变量并运行：

```bash
bash scripts/train_prefix65_4gpu.sh resume
```

恢复记录检查完整配置；仅恢复本次 prefix65 运行产生的 checkpoint。
日志与退出记录位于 RESULT_ROOT 下，模型阶段目录为 main。

## 评价与回传

训练按原 Validation24 规则选优。完成后使用本次 best.pt 评价完整 Validation100：

```bash
python -m cylinderflow evaluate \
  --config cylinderflow_config_prefix65_4gpu.json \
  --dataset "$DATA" --manifest "$MANIFEST" \
  --prepared "$PREPARED" \
  --checkpoint "$RESULT_ROOT/main/best.pt" \
  --mode validation --device cuda:0 --output-dir "$RESULT_ROOT/validation100"
```

回传完整指标、训练曲线、选优记录、运行配置和退出记录。
现有物理指标、预测导出入口可使用本次选中权重，配置指定本页的 prefix65 配置。

交付验证覆盖静态语法、配置、取样范围及 diff 检查。正式四卡环境及训练运行结果由接收端确认。
