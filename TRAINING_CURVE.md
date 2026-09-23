# 历史 checkpoint：Validation100 UV 误差表

**群里统一交接入口：**[四个baseline的运行步骤与图片回传要求](https://github.com/yan-borui/graph-dit-cylinderflow-stride8/blob/main/TRAINING_CURVE.md)。

仅评测已有预测模型，不重训、不更改原选优记录、不访问Test。使用原CUDA推理环境。
行坐标严格为checkpoint内部update × 4，这是本次约定的横轴变换，不等同于所有方法的有效batch或GPU时间。
保持原预测与物理UV相对RMSE定义。

## 运行

先将RUN设为**一个训练run**的绝对目录；自动发现其checkpoints目录中的.pt/.ckpt/.pth；
没有checkpoints目录时扫描RUN本身，不递归搜索其他run。OUT使用新的独立目录。
其他变量沿用原评测配置及prepared数据，AE必须与动力学checkpoint绑定。
Text2PDE的数据位置沿用LDM YAML；RUN只指向LDM run，AE不参与曲线。
本脚本按checkpoint内部真实update排序，重复副本去重；同update不同checkpoint需显式选择。

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
python training_curve.py --run-dir "$RUN" --config "$CONFIG" \
  --dataset "$DATASET" --manifest "$MANIFEST" --prepared "$PREPARED" \
  --output-dir "$OUT"
```

可加 `--checkpoints /absolute/run/checkpoints/a.pt /absolute/run/checkpoints/b.pt` 指定子集；
所有文件必须属于RUN。重复原命令会复用完整成功点，失败/中断点加 `--retry-failed` 重试。
单个checkpoint在独立进程中执行，失败保留日志并继续其他点。
`collector.lock` 防止重复启动；异常强制退出后，确认旧进程已结束才移除残留锁。

## 回传图片

**回传OUT/pages.json列出的全部table_XX.png**。每页最多15行，只有训练投入和UV相对RMSE两列。
每完成一个checkpoint就更新图片；“—”代表缺失、失败或尚未完成，具体原因见status.json。
保留OUT用于续跑；精确分数在status.json，table.csv和PNG展示6位有效数字。
MGN另输出history.csv，保留精确UV分数、真实累计样本数、checkpoint累计耗时、训练GPU数及
累计GPU-hours。累计耗时直接取checkpoint的elapsed_seconds，GPU数来自保存的四卡训练状态，
GPU-hours为二者相乘再除以3600，包含该运行已计入的验证与保存时间。缺少的记录留空。
训练历史的时间与本次重评耗时分别记录，history.csv中的时间对应原训练进度。
底层评价器沿用既有UV指标，逐轨迹物理指标和流场由[PREFIX65.md](PREFIX65.md#论文结果回传)的
所选权重评价入口生成。

本次增加了checkpoint成本元数据。已有旧版曲线输出保留；首次使用新版本请指定新的OUT，
随后可按原命令续跑这个OUT。

## 合并四个方法

四个status.json可在任一仓库执行以下命令，无需加载模型：

```bash
python training_curve.py --merge /results/mgn/status.json /results/eagle/status.json \
  /results/aroma/status.json /results/text2pde/status.json --output-dir /results/merged
```

输出按各方法update×4的并集排列，缺点留空、不插值；列顺序MGN、EAGLE、AROMA、Text2PDE。
可再加入GLaDiT的status.json，追加第五列，GLaDiT坐标不乘4。只收到图片时按显示精度转录，
不补造隐藏小数。每种方法仅合并一个run；不同run分别制作表格。

首次交付进行静态检查。正式GPU评价及失败状态以运行输出为准。
