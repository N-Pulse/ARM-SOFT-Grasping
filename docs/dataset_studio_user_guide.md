# ARM-SOFT Dataset Studio 使用手册

Dataset Studio 是本地运行的 RGB-D 数据与 YOLO Seg 工作台。完整流程为：

```text
Capture → Sessions → Preprocess → Annotate → Review → Dataset → Train → Test
                                                                    │
                                                                    └─ Failure Cases 回流到 Sessions
```

## 1. 启动前准备

1. 进入 `feature/object-segmentation` 工作目录。
2. 激活项目 Python 环境并安装项目依赖。
3. 将 Intel RealSense D405 接入 USB 3 端口。
4. 确认相机没有被 RealSense Viewer 或其他程序占用。

普通启动：

```bash
python tools/dataset_web.py
```

如果 macOS 无法直接访问 D405：

```bash
sudo -E "$(which python)" -u tools/dataset_web.py
```

终端出现以下信息后，在浏览器打开 <http://127.0.0.1:8765>：

```text
[dataset-web] open http://127.0.0.1:8765
```

按 `Ctrl+C` 安全停止服务。不要把终端中的网址当作 shell 命令执行。

无相机时可使用模拟器检查界面：

```bash
python tools/dataset_web.py --source sim
```

## 2. Capture：采集 RGB-D 数据

1. 填写唯一的 `Dataset name`，例如 `box_normal_light_001`。
2. 设置目标帧数和 `Save every N frames`。
3. 设置有效深度范围；当前常用范围是 `0.10–1.50 m`。
4. 在画面中缓慢移动或旋转物体，覆盖不同角度、距离和位置。
5. 点击 `Start capture`，达到目标帧数后自动完成。

每个保存帧包含原始 RGB、对齐的 float32 depth、相机内参和 metadata。连续画面太相似时会自动跳过，避免数据集被重复帧填满。

## 3. Sessions：查看和管理原始数据

- 点击 session 卡片或 `View all photos` 查看全部 RGB 帧。
- `Delete session` 会删除该 session 的原始 RGB-D、derived cache 和 annotations。
- 如果 session 已被 dataset export 引用，系统会阻止删除；应先删除下游内容。

原始数据是不可重建的数据源。只有确认无用时才删除。

## 4. Preprocess：深度与桌面过滤

1. 选择一个 raw session。
2. 填写版本名，例如 `table_filter_v1`。
3. 设置 near/far depth、ROI 和桌面最小高度。
4. 点击运行并等待 `COMPLETE`。
5. 查看 `Raw / Filtered / Mask` 三栏，逐帧确认物体没有被错误切除。

不同参数应使用不同版本名。删除 preprocessing version 只移除可重建的 filtered image、mask 和配置，不删除 raw RGB-D。

## 5. Annotate：生成初始分割标注

1. 选择 input session 和 preprocessing version。
2. 选择 proposal method。
3. 设置 `Class ID` 与 `Class name`。
4. `Min area` 用于过滤很小的噪声区域；`Max instances` 限制单帧候选数量。
5. 点击 `Generate annotations`。

系统在原始 RGB 上保存可编辑 polygon JSON。重新生成时，`Regenerate pending proposals` 只替换尚未审核的 proposal，已批准或拒绝的决定会保留。

当前每个 raw session 只保留一份 canonical annotation set；再次生成会更新该 session 的现有标注，而不是复制整套 RGB 图片。

## 6. Review：人工修正标注

对每一帧进行以下操作：

- 点击彩色 mask 选择物体；
- 修改 class name；
- 删除错误物体；
- 使用 `Draw new object` 绘制缺失物体 polygon；
- `Approve & next`：保存并批准当前帧；
- `Reject frame`：标记该帧不进入训练集；
- `Save changes`：保存修改但不改变审核状态。

只有 `Approved` 标注会进入 Dataset。错误轮廓、背景误检或类别名不正确时不要直接批准。

## 7. Dataset：校验并导出 YOLO Seg 数据集

1. 选择要使用的 approved sessions。
2. 以完整 session 为单位分配 `train / val / test`。
3. 运行 validation，修复所有错误。
4. 输入 export version，例如 `object_seg_v1`。
5. 导出 YOLO segmentation 数据集。

系统生成 `data.yaml`、YOLO labels、images 和 manifest。RGB 图片在同一磁盘上优先使用 hard link，因此 export 不会再次占用一份完整图片空间。不要把相邻帧拆到 train 和 validation，以免数据泄漏。

## 8. Train：训练 YOLO Seg

1. 选择一个通过校验的 dataset export。
2. 输入唯一的 run name。
3. 选择 segmentation 模型，例如 `yolo11n-seg.pt`。
4. 设置 epochs、image size、batch size 和 device。
5. 点击 `Start training`。

页面会显示 epoch 进度、日志和最终 metrics。完成的 run 保存配置、结果图表和 `weights/best.pt`。首次使用预训练 checkpoint 时可能需要联网下载。

## 9. Test：实时测试训练模型

1. 连接 D405，并确认 Train 页面已有完成的 run。
2. 选择 trained model 和 device。
3. 设置 image size、FPS limit 和 confidence threshold。
4. 点击 `Start live test`。

页面显示实时 mask、类别、置信度、推理 FPS、延迟、物体深度以及相机坐标系 XYZ。系统只处理最新摄像头帧，旧帧不会排队，因此不会随着时间产生越来越大的延迟。

### Failure Cases

发现模型错误时：

1. 选择错误类型：`Missed detection`、`False positive`、`Bad mask`、`Wrong class` 或 `Other`。
2. 可在 Note 中说明正确结果。
3. 点击 `Save to failure collection`。

同一次 live test 中保存的案例进入同一个 `failure_...` session。每一帧保留：

- 原始 RGB 和 depth；
- 错误类型与人工备注；
- 当时的模型 run 和预测摘要；
- 相机内参与采集 metadata。

近似重复的相邻失败帧会被跳过。结束测试后，该 collection 会出现在 Sessions，可重新执行 `Preprocess → Annotate → Review → Dataset → Train`，形成主动学习闭环。

## 10. 数据位置与存储规则

所有相关 worktree 默认共享：

```text
../ARM-SOFT-data/
├── raw/object/                     # 唯一不可重建的 RGB-D
├── derived/object/                 # 可重建 preprocessing cache
├── annotations/object_segmentation/# canonical annotation JSON 与 previews
├── exports/object_segmentation/    # 可重建 YOLO datasets
├── runs/object_segmentation/       # 训练结果与 best.pt
└── catalog/                        # Web catalog
```

可用 `ARM_SOFT_DATA_ROOT` 或 `--data-root` 指向其他专用数据目录。不要将 data root 设置为 home directory 或文件系统根目录。

推荐删除顺序：

```text
training run → dataset export → annotation → preprocessing version → raw session
```

删除操作要求输入完整名称。Web 页面默认只监听 `127.0.0.1`，没有认证；不要使用 `--allow-remote` 暴露到不可信网络。

## 11. 常见问题

### 页面打不开或显示 `ERR_CONNECTION_REFUSED`

确认启动终端仍在运行，并使用浏览器打开 `http://127.0.0.1:8765`。如果终端已经回到命令提示符，说明服务没有启动成功。

### `No RealSense device found`

- 使用 USB 3 端口和支持数据传输的线缆；
- 关闭其他占用相机的软件；
- 重新插拔 D405；
- macOS 可尝试上面的 `sudo -E` 启动方式。

### 页面出现旧功能或 API 返回 HTML/JSON 错误

按 `Ctrl+C` 停止旧服务，重新启动 `dataset_web.py`，然后强制刷新浏览器。

### Train 或 Test 没有可选项

- Train 需要先完成 Dataset export；
- Test 需要 Train 成功产生 `weights/best.pt`；
- Test 还需要摄像头在线。

### Permission denied

通常是之前用 sudo 创建的数据仍属于 root。新版 Studio 会在 sudo 采集和 Failure Case 保存后恢复给原登录用户；旧目录可由管理员修正所有权后再使用。
