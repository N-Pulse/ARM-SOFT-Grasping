# N-Pulse 视觉数据与模型训练流程

> 最新进度：本地 Dataset Studio 已连接 Capture、Sessions、Preprocess、Assisted Annotation、逐帧 Review、YOLO-Seg Dataset export、Train 和 Live Test。Dataset 页面只导出 Approved 标注，按完整 session 划分 train/val/test，执行完整性与泄漏检查，并生成 `data.yaml`、labels 和 manifest。Train 页面可选择导出的 dataset、YOLO segmentation checkpoint、epochs、image size、batch 和 device，实时查看日志与进度，并保存每次实验的 metrics 与 best weights。Live Test 页面会读取完成训练的 `best.pt`，对 D405 最新帧做低延迟分割，同时显示置信度、推理速度、物体深度和 XYZ。测试者可以给失败案例选择 `missed detection`、`false positive`、`bad mask`、`wrong class` 或 `other`，补充备注，并将同一次测试中的案例集中保存回一个 Sessions 数据集；近似重复帧会自动跳过。
>
> 数据架构更新：所有 worktree 默认共享 `CV_Project_V2526/ARM-SOFT-data`。原始数据、derived cache、annotations、exports 和 models 分层保存，避免 branch 间复制数据。采集器会跳过高度相似的连续帧。

## 1. 最终目标

建立一条可以重复运行的 RGB-D 视觉流水线，最终输出：

1. **物品实例分割**：识别可抓取物品并生成精确 mask；
2. **机械手姿态估计**：用自定义 YOLO Pose 模型输出机械手/假肢的关键点；
3. **3D 信息**：将 2D mask 或关键点与对齐 depth、相机内参结合，计算相对摄像头的距离和 3D 坐标；
4. **抓取逻辑**：把物品和机械手的 3D 结果交给现有 grasping pipeline。

人物全身和普通人体姿态识别不在当前目标内。MediaPipe 只保留为人体手臂实验，不作为机械手主模型。

## 2. 总体架构

```text
RealSense D405 / simulator / replay
                  │
                  ▼
       shared RGB-D collector
       raw RGB + raw depth + intrinsics + metadata
                  │
          ┌───────┴────────┐
          ▼                ▼
 shared depth filter   immutable raw dataset
 (derived mask only)        │
          │          ┌──────┴──────────────┐
          │          ▼                     ▼
          │   object segmentation   mechanical-hand pose
          │   assisted annotation   assisted annotation
          │   SAM/existing model    orange-tip tracker
          │          │                     │
          │     YOLO-Seg dataset       YOLO-Pose dataset
          │          │                     │
          └──────────┴──────────┬──────────┘
                                ▼
                         training/evaluation
                                │
                                ▼
                   RGB-D inference and 3D projection
```

## 3. 共享采集数据契约

每次采集产生一个独立 session：

```text
datasets/captures/<session>/
├── session.json       # schema、用途、内参、单位、帧数、备注
├── manifest.jsonl     # 每帧的文件路径、时间、来源、有效 depth 比例
├── color/
│   └── frame_000000.png
└── depth/
    └── frame_000000.npy   # float32，单位 metre，与 RGB 像素对齐
```

规则：

- 原始 RGB、depth、内参和 metadata 必须一起保存；
- 原始数据不可被 preprocessing 或 annotation 覆盖；
- session 内相机内参不能变化；
- 数据集划分以 **session** 为单位，避免连续相邻帧同时进入 train 和 val 造成数据泄漏；
- 大型图片、depth、labels 和 weights 不提交 Git；Git 只管理代码、配置、文档和小型 manifest。

### 采集命令

```bash
# 接 RealSense 后采集 300 帧机械手数据
python tools/collect_rgbd_dataset.py \
  --source camera --purpose mechanical-hand-pose --frames 300 \
  --out datasets/captures/hand_session_001 \
  --notes "orange fingertips, frontal view"

# 每 5 帧保存一帧，减少高度重复的数据
python tools/collect_rgbd_dataset.py \
  --source camera --purpose object-segmentation --frames 200 --every 5

# 无相机时验证采集模块
python tools/collect_rgbd_dataset.py \
  --source sim --purpose general --frames 5 --out /tmp/armsoft_capture_test
```

## 4. Depth preprocessing 应该放在哪里

Depth filtering 是两个任务共用的基础能力，所以放在：

```text
armsoft/preprocessing/depth_filter.py
```

而不是放进 object model 或 mechanical-hand model。它接收原始 RGB 和 depth，返回：

- 指定距离范围内的 boolean mask；
- 将范围外像素置为背景的 RGB 副本。

建议第一版根据工作空间设置 `min_depth_m` 和 `max_depth_m`。后续可添加桌面平面消除、ROI、多帧 temporal filtering，但必须保留原始 session，通过配置可重复生成结果。

当前 depth preprocessing 分支已经支持三个可以组合的过滤条件。Web 中 Depth 始终启用，ROI 与 table plane 可分别开启或关闭：

1. near/far distance：保留指定相机距离；
2. normalized ROI：用 `x1,y1,x2,y2` 限制画面工作区域；
3. table plane：优先从整个 session 的多帧估计稳定的 `nx,ny,nz,d`，再用最小离桌高度去掉桌面及桌面下方点。整体估计失败时按“当前帧拟合 → 上一帧平面 → 仅 Depth + ROI”降级，不删除原始帧。

先用三栏实时画面检查参数（RAW / FILTERED / MASK，按 `Q` 退出）：

```bash
python tools/preview_depth_filter.py --source camera \
  --min-depth 0.10 --max-depth 0.80 --roi 0.1,0.1,0.9,0.9
```

处理一个已经采集的 session：

```bash
python tools/preprocess_rgbd_session.py datasets/captures/hand_session_001 \
  --out datasets/captures/hand_session_001/derived/depth_v1 \
  --min-depth 0.10 --max-depth 0.80 \
  --roi 0.1,0.1,0.9,0.9
```

如果已经通过棋盘标定得到平面 `normal` 和 `d`，可进一步去掉桌面：

```bash
python tools/preprocess_rgbd_session.py datasets/captures/hand_session_001 \
  --out datasets/captures/hand_session_001/derived/table_v1 \
  --table-plane 0,-1,0,0.08 --min-table-height 0.005
```

输出目录包含 `config.json`、逐帧 binary masks、暗化背景的 previews 和可追溯到 raw session 的 `manifest.jsonl`。这些都是 derived artifacts，删除后可以根据配置重建。

## 5. 两个数据集工作流

### A. 物品 YOLO Segmentation

1. 用共享 collector 在多物品、角度、距离、背景和光照条件下采集；
2. 用 depth filter 去掉工作距离外背景；
3. 用 SAM/SAM2 或已有 segmentation model 生成初始 mask；
4. 人工检查和修正边界、漏检和错检；
5. 导出 YOLO segmentation 格式；
6. 按 session 划分 train/val/test，训练 YOLO-Seg；
7. 在 test sessions 上测 mAP、mask IoU，并额外测试遮挡和相似背景；
8. 合格模型用于下一轮 pseudo-label，人工只修正低置信度样本。

当前分支已经实现第一个可运行闭环。Depth mask 的 connected components
提供候选实例框；加上 `--sam-model mobile_sam.pt` 后，候选框会作为 SAM box
prompts 精修边界。内部 JSON 保存像素 polygon、来源和审核状态，只有审核通过的
标注默认会进入 YOLO 数据集。

```bash
# 1. 生成 proposal；不指定 SAM 时只使用 depth components
python tools/annotate_object_segmentation.py datasets/captures/object_session_001 \
  --derived datasets/captures/object_session_001/derived/depth_v1 \
  --out datasets/object_segmentation/object_session_001 \
  --class-id 0 --class-name object --sam-model mobile_sam.pt

# 2. 快速审核：A 接受，R 拒绝，S 跳过，Q 退出
python tools/review_object_annotations.py \
  datasets/object_segmentation/object_session_001

# 3. 按 session 导出。不同 session 分配到 train/val/test，不能拆连续帧
python tools/export_yolo_segmentation.py \
  datasets/object_segmentation/object_session_001 \
  --out datasets/object_segmentation/exports/yolo_v1 --split train

# 4. 训练并保留训练来源、参数和 best weights 路径
python tools/train_object_segmentation.py \
  --data datasets/object_segmentation/exports/yolo_v1/data.yaml \
  --model yolo11n-seg.pt --epochs 100 --device cpu --name yolo_seg_v1
```

对于需要修改 polygon 边界的样本，JSON 是 annotation source of truth；可以接入
CVAT/Label Studio 转换器。当前键盘审核器只负责接受/拒绝 proposal，不冒充完整的
polygon editor。

建议结构：

```text
datasets/object_segmentation/
├── annotations/       # annotation tool 的项目/中间结果
├── splits/            # 固定的 session 划分清单
└── exports/yolo/      # images、labels、data.yaml
```

### B. 机械手 YOLO Pose

1. 采集不同关节角度、距离、视角、背景、光照和遮挡；
2. depth filter 限制工作空间；
3. 现有橙色指尖 tracker 自动给出指尖候选；
4. 人工补齐/修正其余机械手关键点和可见性；
5. 导出统一 17 点定义的 YOLO Pose 数据；
6. 训练第一版 YOLO Pose；
7. 用第一版模型 pseudo-label 新数据，再人工校正；
8. 将 2D keypoint 对应到 depth；邻域 depth 无效时使用中位数/最近有效像素，并输出 3D 置信度。

建议结构：

```text
datasets/mechanical_hand_pose/
├── annotations/
├── splits/
└── exports/yolo/
```

## 6. Branch / workflow 规划

```text
independent-cv                         稳定集成基线
├── feat/rgbd-dataset-collector        本分支：数据契约、采集器、目录骨架
├── feat/depth-preprocessing           深度 ROI/桌面过滤及可视化
├── feature/object-segmentation        物品标注、YOLO-Seg 训练与推理
├── feature/mechanical-hand-pose       橙色指尖、17 点标注、YOLO-Pose
└── feat/rgbd-pose-integration         成熟后整合两个模型与 3D 输出
```

Branch 是一条代码开发线；workflow 是从采集、标注、训练、评估到部署的一系列步骤。一个 workflow 会跨越多个 branch，branch 不应该被当成数据或执行流程。

合并顺序：共享 collector → depth preprocessing → 两个模型可并行开发 → integration。每个功能分支通过测试和小型真实数据 smoke test 后才合并到 `independent-cv`。

## 7. 实施阶段与验收标准

### Phase 1 — 采集基础设施（当前）

- 相机/sim/replay 都能输出同一 session 格式；
- RGB 和 depth 对齐且不被修改；
- session metadata 足以重放并追踪来源；
- 至少在 simulator 上自动测试通过。

### Phase 2 — Depth preprocessing

- 配置化 near/far clipping；
- 保存/显示 mask，但不覆盖 raw data；
- 在真实 D405 样本上检查机械手和物品未被切掉；
- 后续加入桌面平面和工作区 ROI。

### Phase 3 — Assisted annotation

- 物品：自动 mask + 人工修正 + YOLO-Seg export；
- 机械手：橙色指尖候选 + 17 点人工修正 + YOLO-Pose export；
- annotation 工具保存版本和可见性，不只保存渲染后的图片。
- **当前状态**：物品 mask proposal 的 Web 批处理、进度、结果预览和逐帧 Review 已完成；重新生成只覆盖 pending proposal，保留 approved/rejected 决定。Review 可重命名、删除或手动画 polygon，并重新生成 preview。
- Annotation 也支持 `Raw RGB-D` bypass：MobileSAM 可直接在原始 RGB 上自动提议 mask，或创建空标注供 Review 手绘；该路径不创建 preprocessing 副本。

### Phase 4 — Model training

- 数据按 session 固定划分；
- 训练配置、类别/关键点 schema 和模型版本进入 Git；
- weights 存 release/artifact storage，不直接进入 Git；
- test set 在训练期间保持冻结。
- **当前状态**：YOLO-Seg dataset 的 Web 校验、版本化导出和删除已完成。导出优先对 raw RGB 使用硬链接以避免重复存储，删除 export 不影响 raw data 或 canonical annotations。Train 页面已连接 YOLO segmentation 训练、实时日志、epoch 进度、metrics、best weights、停止任务和实验删除。Live Test 已连接完成的训练结果与 D405 RGB-D 流，并支持将误检或漏检现场作为新的单帧采集 session 回流。

### Storage 与删除规则

- `raw/` 是唯一不可重建的数据源，RGB 与 depth 各保存一次；
- `derived/` 和 `annotations/*/previews/` 是可重建缓存，后续生成使用最高 PNG 压缩；
- raw session 内的 `derived` 是软链接，不产生第二份数据；
- YOLO export 的 RGB 优先使用硬链接，同一磁盘上不重复占用图片空间；
- Sessions 页面分别显示 raw、preprocess cache、annotation cache、exports 和 training runs，并按 inode 计算真实物理占用；
- 删除操作均要求输入完整名称。正在运行的任务及被下游 export 引用的数据禁止删除；应按 `training run → dataset export → annotation → preprocessing → raw session` 的逆序清理。

### Local Web 安全边界

- 默认只监听 `127.0.0.1`；非本机监听必须显式添加 `--allow-remote`，且当前没有用户认证；
- API 拒绝跨域写请求和超过 1 MiB 的请求体，并发送 CSP、禁止 iframe 和 MIME sniffing 等浏览器安全头；
- 所有文件标识都经过白名单和真实路径边界检查，subprocess 使用参数数组而不是 shell；
- 服务退出时会终止预处理、标注和训练子进程，避免孤立后台任务。

### Phase 5 — RGB-D integration

- segmentation mask 和 pose keypoints 映射到可靠 depth；
- 输出相机坐标系中的距离/3D 坐标、置信度和失败原因；
- 实时性能和错误安全策略通过实机测试。

## 8. 采集覆盖清单

每个数据集都应覆盖：不同距离、俯仰/旋转角度、左/右画面位置、光照、复杂背景、运动模糊、部分遮挡和 depth 缺失。机械手另外需要覆盖不同手指组合与橙色指尖被遮挡；物品另外需要覆盖多个实例接触、透明/反光表面和不同尺寸。

在第一轮真实采集后再锁定类别表、17 点最终定义、工作距离和评价阈值，避免在没有样本证据时过早固定。
