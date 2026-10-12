# N-Pulse 视觉数据与模型训练流程

## 当前进度（2026-10-13）

| 阶段 | 状态 | Branch | 已实现内容 |
| --- | --- | --- | --- |
| RGB-D 数据采集 | ✅ 完成 | `feat/rgbd-dataset-collector` | versioned session、RGB/depth/内参/manifest、覆盖保护 |
| Depth preprocessing | ✅ 完成 | `feat/depth-preprocessing` | near/far、ROI、桌面平面、实时预览、derived export |
| 物品 segmentation | ✅ 第一版闭环完成 | `feature/object-segmentation` | depth proposals、可选 SAM、审核、YOLO-Seg export/train |
| 机械手 pose | ✅ 第一版标注/训练闭环完成 | `feature/mechanical-hand-pose` | 橙色候选、17 点编辑、YOLO-Pose export/train |
| 模型质量提升 | ⏳ 待真实数据 | 各模型分支 | pseudo-label、hard cases、冻结 test set、指标门槛 |
| RGB-D 3D integration | ⏳ 下一阶段 | `feat/rgbd-pose-integration` | 2D→depth→3D、置信度、实时联合推理 |

“完成”表示代码路径与模拟/单元测试可运行，不表示模型已经训练完成。真正的模型精度取决于真实 D405 数据、人工审核质量、session-level split 和独立 test set。

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

当前 depth preprocessing 分支已经支持三个可以组合的过滤条件：

1. near/far distance：保留指定相机距离；
2. normalized ROI：用 `x1,y1,x2,y2` 限制画面工作区域；
3. table plane：用 `nx,ny,nz,d` 和最小离桌高度去掉桌面及桌面下方点。

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

当前机械手分支已经实现前 1–6 步的工具链。橙色 tracker 只产生无语义的
fingertip candidates；标注者仍确认每个指尖身份，点击时会自动吸附到附近候选点。
这样不会因手掌旋转而错误地把“从左到右”当成“拇指到小指”。

```bash
# 1. 建立 17 点模板并检测橙色候选（无 GUI，适合先批量运行）
python tools/annotate_mechanical_hand_pose.py datasets/captures/hand_session_001 \
  --derived datasets/captures/hand_session_001/derived/depth_v1 \
  --out datasets/mechanical_hand_pose/hand_session_001 --bootstrap-only

# 2. 交互标注；左键=可见，右键=遮挡，x=不存在，u=返回，a=批准
python tools/annotate_mechanical_hand_pose.py datasets/captures/hand_session_001 \
  --derived datasets/captures/hand_session_001/derived/depth_v1 \
  --out datasets/mechanical_hand_pose/hand_session_001

# 3. 以 session 为单位导出 train/val/test
python tools/export_yolo_pose.py datasets/mechanical_hand_pose/hand_session_001 \
  --out datasets/mechanical_hand_pose/exports/yolo_pose_v1 --split train

# 4. 训练自定义 17 点模型
python tools/train_mechanical_hand_pose.py \
  --data datasets/mechanical_hand_pose/exports/yolo_pose_v1/data.yaml \
  --model yolo11n-pose.pt --epochs 150 --device cpu
```

导出的 YAML 使用 Ultralytics 的 `kpt_names` 结构，并显式设置机械手自己的
`kpt_oks_sigmas`。因为本项目也恰好有 17 点，如果不设置，框架可能使用 COCO
人体 17 点的默认 OKS 权重，而这些语义与机械手 skeleton 不一致。

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

### Phase 4 — Model training

- 数据按 session 固定划分；
- 训练配置、类别/关键点 schema 和模型版本进入 Git；
- weights 存 release/artifact storage，不直接进入 Git；
- test set 在训练期间保持冻结。

### Phase 5 — RGB-D integration

- segmentation mask 和 pose keypoints 映射到可靠 depth；
- 输出相机坐标系中的距离/3D 坐标、置信度和失败原因；
- 实时性能和错误安全策略通过实机测试。

## 8. 采集覆盖清单

每个数据集都应覆盖：不同距离、俯仰/旋转角度、左/右画面位置、光照、复杂背景、运动模糊、部分遮挡和 depth 缺失。机械手另外需要覆盖不同手指组合与橙色指尖被遮挡；物品另外需要覆盖多个实例接触、透明/反光表面和不同尺寸。

在第一轮真实采集后再锁定类别表、17 点最终定义、工作距离和评价阈值，避免在没有样本证据时过早固定。
