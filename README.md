# adkit — 只用良品做缺陷检测

无监督缺陷检测框架：模型只学习「良品长什么样」，推理时输出像素级异常分数 + 缺陷分类。
目标缺陷：**裂纹 / 脏污 / 缺失**。

## 架构

```
原图 → ROI 定位 → 视角分流 → 规范化(旋转/缩放/翻转/平移对齐) → [检测器组] → 归一化融合 → 阈值判定 → 连通域分类 → 报告
```

规范化是整套系统的地基。所有检测器都只看规范化后的 256×256 图，异常图再通过单应矩阵映射回原图坐标。

| 层 | 位置 | 内容 |
|---|---|---|
| 预处理 | `src/adkit/preprocess/` | `roi.py` Otsu+轮廓定位 · `canonical.py` 单应对齐 · `view.py` 视角分流 |
| 检测器 | `src/adkit/detect/` | `patchcore.py` `stefann.py`（特征近邻/马氏距离）· `crack.py` `missing.py` `stain.py`（经典 CV） |
| 后处理 | `src/adkit/postprocess/` | `threshold.py` 良品分布标定 · `fuse.py` 融合 · `classify.py` 缺陷分类 · `viz.py` 可视化 |
| 编排 | `src/adkit/pipeline.py` `cli.py` `evaluate.py` `data/dataset.py` | |

## 检测器

**PatchCore**（首选）— 冻结 CNN 提 patch 特征，良品特征建 memory bank（可选 FPS coreset 降采样），推理时算 kNN 距离。没有梯度训练，几十张良品即可起步，CPU 可跑。

**STEFANN** — 逐网格位置的特征均值 + 收缩协方差，马氏距离出异常图。定位比 PatchCore 锐利，但要求对齐严格、每个视角至少 `min_images`（默认 4）张良品。

**crack** — Steger Hessian 线度量（多尺度取最大）作用在「当前图 − 良品中位图」的残差上，再除以**纯配准偏移所能产生的响应**（fit 时把参考图平移 ±`jitter_px` 逐像素量出来），最后做细长形状过滤。除以噪声底而不是屏蔽强梯度附近的整片区域：后者在焊道盖板上会瞎掉 87% 的画面，而划痕恰恰沿盖缝走。

**missing** — 良品轮廓（膨胀/腐蚀容差）与当前轮廓的差集：`lost`=缺料，`extra`=多余料/毛刺。**已知局限**：两侧的掩膜目前都来自拟合矩形，矩形对矩形看不见被切掉的角，所以缺料在像素层面不响应（判定仍可能由其他通道给出）。要修得先有可靠的零件自身轮廓提取，边缘闭运算那条路已实测失败（详见任务 #14 结论）。

**stain** — 与良品中位图的残差，按良品自身波动归一化成 z 分数，滤波后取面积。

所有检测器实现同一个 `Detector` 接口（`fit` / `score_map` / `save` / `load`），输出 `ScoreMap`。

## 安装

torch 的 DLL 在含非 ASCII 的路径下加载会失败，所以 **venv 放在 ASCII 路径**，项目本身可以在中文路径。

```bash
# <VENV> is any ASCII path, e.g. D:/venvs/adkit or $HOME/.venvs/adkit
uv venv <VENV> --python 3.11
uv pip install --python <VENV>/Scripts/python.exe numpy scipy \
    opencv-python-headless scikit-image scikit-learn pyyaml click pillow
uv pip install --python <VENV>/Scripts/python.exe torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
```

第二步必须单独跑：`--index-url` 会覆盖 PyPI 源，合在一条命令里会导致 numpy/scipy 找不到。

## 用法

```bash
export ADKIT_PY=<VENV>/Scripts/python.exe
$ADKIT_PY -m adkit.cli status

# 1. 分拣：规范化每张图、猜视角、出缩略图墙和 manifest
$ADKIT_PY -m adkit.cli prepare image/

# 2. 人工把图挪进 data/<view>/good 或 data/<view>/ng
#    ng 下再按类型建子目录（ng/crack、ng/stain、ng/missing）可验证分类准确率

# 3. 建模 + 用良品分数分布标定阈值
$ADKIT_PY -m adkit.cli fit --detectors patchcore,stefann,crack,missing,stain

# 4. 推理
$ADKIT_PY -m adkit.cli predict image/new.png --out runs/infer

# 5. 用留出的 ng 集验证召回率和过检率
$ADKIT_PY -m adkit.cli evaluate
```

## 配置

`config/default.yaml`，重点：

- `preprocess.canonical.size` — 规范化画布尺寸。改它等于换一套模型，必须重新 `fit`。
- `preprocess.view.assignment` — `manual` 按目录名分视角；`auto` 按 `bbox_aspect / mean / fill / edge` 规则分流。
- `detectors.*.grid` — 异常图分辨率。裂纹很细，`patchcore.grid` 建议 ≥ `[32,32]`。
- `detectors.*.pretrained` — 设 `false` 可离线跑（精度会明显下降，仅用于打通流程）。
- `calibration.pixel_percentile` — 良品像素分数的分位点，即融合图上的 `1.0`。调大 = 更严（漏检多），调小 = 过检多。

## 阈值语义

`fit` 阶段在每个视角上把所有良品过一遍全部检测器，取像素分数 `pixel_percentile` 分位作为该检测器的 `hi`、中位数作为 `lo`。融合图因此归一化到 0..1，**1.0 就是「良品里最极端的像素」**。所以：

- 图像阈值 = 各检测器良品参考值 × `image_margin`
- 像素阈值 `pixel_cut` 不是拍的：fit 时把每张良品在 `region_probe` 分位上的区域峰值量出来，再乘裕度（下限 `fusion.pixel_cut`），所以它会随这批良品的成像质量走

在良品上完全沉默的通道（比如没有裂纹的良品集上的 crack），它的参考值会塌到 0，此时改用该通道自己的绝对像素参考（crack 用标定出的线阈值、stain 用 z 门限），否则「除以 0 附近的地板」会把任何非零读数都读成十倍超标。

## 判决线取分位数，不取最大值

三条线（图像级 `image_ref`、像素级 `pixel_cut`、面积级 `extent_cut`）都取**逐张良品最差值的 p99 分位数**（`calibration.reference_percentile`），不是这批良品的最大值。

原因不是理论洁癖，是实测出来的：最大值随批次单调上升，所以**良品越多系统越钝**。209 张良品时像素线涨到 2.66、面积线 0.0551，而真正没见过的良品只到 1.24 / 0.0017 —— 一张真实的破口（峰值 1.68、面积证据 0.0068）因此以 1.14 倍被判成良品（判决线 1.15），且一个框都没有。改成 p99 后：像素线 2.06、面积线 0.0050，63 张带标注不良的判决命中从 **41/63 升到 61/63**，未见良品误判仍是 **0/20**；14 张良品的真图完全不变（小样本下 p99≈max），3/14 误判、16/16 命中。

代价是**明说**的，不是藏起来的：分位数意味着约 `100 - reference_percentile`% 的良品会越线，fit 会报 `decision lines set at p99; N/209 of these good images already cross them`，并附上批次最大值与 p99 两个数。想要零停线就把该值调到 100，代价是漏检——这次是把代价从暗处搬到明处，不是把它消掉。

剩下没被检出的不良是物理限制：实测有两张污染的面积证据 0.0015 / 0.0021，与未见良品的最大 0.0017 同级，任何线都分不开。

## 样本外验证

`fit` 报的「良品 0 误判」是样本内数字，几乎没有信息量。两层原因：

1. 判决线取的就是这批良品自身分数的分位数，10 张样本上的 p99.9 约等于它们的最大值；一张没见过的良品超过「10 次里的最大值」差不多是对半概率。
2. 学习通道（PatchCore/STEFANN）的样本内分数**结构性偏低**：查询图自己的 patch 就在 memory bank 里，自己的特征也在按位置估计的均值/协方差里。它量到的那个距离不属于「没见过的零件」的分布，所以分位数取多高都不对。

所以标定时每张良品按**留一法**评分（`Detector.calibration_map`：PatchCore 排除属于该图的那些 bank 行，STEFANN 用 rank-1 降级重算该位置的均值与协方差，与「拿掉这张图重拟合」逐元素相等，见 `tests/test_detectors.py`）。推理仍走 `score_map`，改的只是阈值从哪个分布取。

固定相机基准集上的实测（未见良品 ncc 0.834，训练良品 0.851 —— 配准不是变量）：

| 拟合良品数 | 留一法之前的样本外误判 | 之后 |
| --- | --- | --- |
| 20 | 15/20（stefann×12、patchcore×10、missing×3） | **3/20**（三票全来自 missing） |
| 50 | — | **0/20**，最差未见良品 0.96×，参考值终于盖住了未见零件 |

像素级 AUROC 不变（patchcore 0.92–0.98、fused 0.91–0.98），因为推理路径没动。

真图（14 张良品，良品间 ncc 只有 0.673）：折外误判 **7/14 → 2/14**，剩下的两票在 stain/missing，学习通道不再投票。crack 一直是 0/14 —— 它的阈值本来就按实测配准抖动能产生的响应上界来定，不取良品分布的分位数。

fit 同时跑一遍折内协议（`calibration.min_images_per_fold` 决定折数，让每折拟合规模接近上线规模），把**每个良品都被恰好一个折模型评过**的结果报成 `样本外误判 x/y`；UI 门控行和 `anchor.json` 都带这个数，`update()` 用 `_worse_holdout` 按率交叉相乘比较候选与锚点，误判率上升或样本外召回下降都拒绝发布。

```bash
# 真图：k 折留出，报每个通道的误判归属
$ADKIT_PY tools/holdout_eval.py --view tight --ng image/NG --folds 4 --detectors crack
# 基准集：官方 train/test 划分 + 像素 GT，报 AUROC
$ADKIT_PY tools/mvtec_eval.py --dataset bottle --sizes 20,50
```

`bottle/` 是 MVTec-AD 风格的瓶子基准集（209 张 train 良品、20 张 test 良品、63 张带像素掩码的不良图，900×900）。授权 CC BY-NC-SA：**只能用于本地验证，不能进产品或商用数据**。它存在的意义是回答真图回答不了的问题——成像可重复、良品足够多时，才能把「统计量错了」和「数据不够」分开的。

**不要把它拷进 `data/<view>/good`。** `data/` 是某个产品的良/不良库，一个视角混进两种产品，每个产品都会被另一种判成缺陷（fit 的告警会说 `n/N good images sit apart from the rest`）。`tools/mvtec_eval.py` 直接读 MVTec 的 `train/test/ground_truth` 目录结构，不需要任何拷贝或建视角。


## 区域判决的两条线

`classify` 把像素图变成"这里有缺陷"，靠两条互相独立的证据，任一成立即算：

| 判据 | 含义 | 门限 |
| --- | --- | --- |
| 峰值 `strong` | 区域内有像素明显超过良品最极端的那个像素 | `pixel_cut` = 良品最大区域峰值（留一法评分）× `image_margin`，下限 `fusion.pixel_cut` |
| 面积 `extensive` | 一大片区域都**略微**超过良品极端值 | `extent_cut` = 良品最大 `mass` × `image_margin`，`mass = Σ(区域内地图像素 map-1.0) / 零件面积` |

面积这条是实测逼出来的。基准集上一张污染覆盖零件约 1/10，峰值只有 1.30，而峰值线在 2.43 以上——旧判据永远过不了线；它的面积证据却是好零件最高值的 18 倍。

两个配套的细节：

- **邻近证据先连通再量**（`classify.merge_px`，默认 3）。污染是颗粒状的：整张图的面积证据 0.0065 已经越过门限 0.00519，但没有任何单独一颗碎斑过线，于是缺陷存在却报不出来。合并后该类的判决从 5/8 升到 7/8。**但送出去给复核员看的 mask 只收真正超过阈值的像素**，膨胀出来的晕不算证据——mask 是复核时逐像素核对的那个声明，把框撑大等于教回流环节接受错误的边界。
- **贴边的区域不再整条丢弃**。旧代码只要区域 bbox 碰到画布边缘就丢；对"零件基本填满画布"的产品这等于关掉整个像素路径（基准集 63 张带掩码的不良图返回 0 个区域，其中一张峰值 4.01、门限 2.59）。现在只丢完全落在最外圈 band 内的条状区域（那是抠图接缝）。

留出实测还报出"折外零件能冲到多高的峰值/面积"，但这两个数**不用来定门限**：折模型训练图更少，样本外峰值被抬高（实测门限从 1.80 被推到 3.00、面积从 0.00816 推到 0.029，而真正没见过的零件只到 1.24 / 0.0082）。用它定门限在真图上直接丢掉两个已知缺陷（16/16 → 14/16）。所以门限来自拟合自己的留一分布，折外只负责报告它成立到什么程度。

`classify.extent_evidence: false` 可以关掉面积判据（只留峰值），这是产线决策不是算法决策：见下表。

## 区域判据改动的实测代价

同一套标定/判决代码，两个数据集：

| 指标 | 改动前 | 改动后 |
| --- | --- | --- |
| 基准集 50 良品：未见良品误判 | 0/20 | 0/20 |
| 基准集：broken_large / broken_small / contamination 判决命中 | 8/8 · 7/8 · 5/8 | 8/8 · **8/8** · **7/8** |
| 真图（14 良品，4 折留出）：未见良品误判 | 2/14 | **3/14** |
| 真图：已知不良命中 | 15/16 | **16/16** |

灵敏度在两个数据集上都涨了；真图多一次停线（14 张里第 3 张）。关掉 `extent_evidence` 就回到 2/14 与 15/16。

剩下那两张 contamination 里有一张（003）是**物理不可分**的：它的面积证据 0.0018 恰好等于未见良品的最大值 0.0018，峰值 1.14 也低于未见良品的 1.24——任何把这两张分开的门限都会先把好零件判死。这不是调参问题。

## 环境：torch 依赖问题的定位与修复（已完成）

torch 装得上但 `import torch` 报：

```
OSError: [WinError 126] ... Error loading ".../torch/lib/shm.dll" or one of its dependencies
```

用 PE 导入表逐个解析后定位到：`torch_cpu.dll` 依赖 **`VCRUNTIME140_THREADS.dll`**，本机没有。
本机装的是 VC++ 运行库 **14.34.31931.0**，而这个 DLL 是 **14.36（VS2022 17.6）才拆出来单独发布的**，
所以 `msvcp140.dll` / `vcruntime140.dll` / `vcruntime140_1.dll` 全都在，只差这一个。
VS2019 装的运行库更老（14.2x），同样不含它——**装了 VS 不等于有这个文件**。

修复（不动系统，只影响本项目）：从微软官方 `VC_redist.x64.exe` 里切出内嵌 cabinet，
取出签名为 Microsoft、版本 14.44.35211.0 的 x64 DLL，放进 venv 的 `torch\lib\`：

```bash
# 1. 下载官方 redist（aka.ms 直链）
curl -L -o VC_redist.x64.exe https://aka.ms/vs/17/release/vc_redist.x64.exe
# 2. 扫出内嵌 cabinet（MSCF 魔数），cbCabinet 长度字段在偏移 +8
# 3. 7z 解出主 cabinet -> 里面的 a12 是 x64 MSI，含 vcruntime140_threads.dll_amd64
# 4. 重命名后拷入 <venv>/Lib/site-packages/torch/lib/
```

如果不想走这条路，直接安装最新版 VC++ 2015-2022 x64 运行库也可以（向后兼容，需要管理员权限）。

另外 venv 必须放在纯 ASCII 路径（例如 `D:/venvs/adkit`）；项目本身在中文路径没问题，
但 `cv2.imread` / `cv2.imwrite` 处理不了中文路径，所以 `imaging.py` 全部走
`np.fromfile` + `cv2.imdecode` / `cv2.imencode` + `tofile`。

## 在当前 27 张图上的实测结果

`fit --detectors patchcore,stefann,crack,missing,stain` + `predict image` 全链路跑通：

- 5 张训练良品全部判为 GOOD，分数 0.92–1.00（图像阈值 1.15）
- 其余 22 张全部 NG —— 这是**预期结果**，因为它们和那 5 张不是同一个视角/缩放，
  `fit` 的告警已经明确说了 `1/5 align poorly`、`only 5 good images (< 30)`
- 缺陷分类输出 `missing` 52 处 / `stain` 13 处 / `crack` 2 处（同样受视角混杂影响，数字本身没有意义）

结论：框架本身可用，**瓶颈完全在数据**——需要按视角各攒 50+ 张良品。

## 在当前 27 张图上 `prepare` 的实测结果

`runs/prepare_manifest.csv` 把视角结构直接量化出来了：

| 簇 | area_ratio | fill | bbox_aspect | 说明 |
|---|---|---|---|---|
| 3 张 PNG | 0.24–0.43 | 0.73–0.94 | ~1.0 | 另一种来源，产品占满画面 |
| `_50/_51/_52/_58/_59` | 0.0097–0.0146 | 0.95–0.96 | 0.81–0.86 | 同一面，缩放一致 |
| `_55/_56/_57/_60/_61/_62` | 0.041–0.043 | 0.89–0.94 | 0.95–1.0 | 同一面，放大一档 |
| `_44/_48/_49` | 0.024–0.025 | 0.53–0.66 | 0.84–1.01 | 又一个面 |

**关键：这个零件接近正方形**（bbox_aspect 0.81–1.04）。`minAreaRect` 在这种轮廓上无法稳定区分长短边，
图与图之间会随机跳 90°，导致对齐崩掉（实测最差 ncc=0.124）。
所以 `canonicalize` 按轮廓长宽比分两条路走：明显长条的用几何（只剩 180° 歧义），
接近正方形的**让内容在 4 个 90° 旋转里挑最优**（`canonical.elongated_aspect` 控制分界）。
改完后同一簇内 5 张图只有 1 张对齐偏弱（ncc 0.802，那张缩放差了一档）。

## Web 复核界面

```bash
$ADKIT_PY -m adkit.cli serve --port 8765
# 浏览器打开 http://127.0.0.1:8765
```

代码在 `src/adkit/ui/`：`app.py` 是 FastAPI 端点，`render.py` 负责把结果画成 PNG，
`static/index.html` 是单页前端（无构建步骤，纯原生 JS）。

界面按实际工作流排：

- **图片列表打开即出** — 不需要先扫描；未打分的行标「未扫」，点某行才真正检测并渲染
- **选择目录** — 弹窗式文件浏览器，从项目根起步，点 `..` 可逐级上钻到任意盘符；每层显示
  「本层图片 N 张」，没有图片的目录「选用这个目录」置灰。目录输入框也能直接粘绝对路径
- **① 分拣 prepare** — 把目录里每张图规范化、猜视角，右侧表格直接列出 `aspect / 面积比 / fill / 均值 / 边缘密度`，
  分视角该不该拆、怎么拆，看这张表就能定
- **归入良品 / 归入不良品** — 复核是**复制**到 `data/<view>/good|ng/<type>/`，原图不动；
  不良品按裂纹/脏污/缺失分子目录，`evaluate` 会据此算分类准确率
- **② 建模 fit** — 首次会下载骨干权重；对齐不达标的图会在界面上直接黄底警告
- **③ 全量扫描** — 整目录打分并按分数倒排，NG/GOOD 徽章一目了然
- **判定线滑杆** — 拖动即时重判全部图片，不用重新建模。用来找过检率和漏检率的平衡点
  （实测：margin 1.15 → 22 张 NG，3.5 → 12 张）
- **四视图** — 原图 / 规范化 / 热图 / 缺陷框；点某行时按需渲染 PNG，不是一次全出
- **缺陷明细** — 每处缺陷带原图坐标系的放大裁剪、面积/细长比/最长尺寸、以及是哪些检测器贡献的

接口：`/api/status` `/api/browse` `/api/images` `/api/dataset` `/api/prepare` `/api/fit` `/api/predict`
`/api/sweep` `/api/review` `/api/artifact/{token}/{name}`。

### 目录访问范围

`config/default.yaml` 的 `ui` 段控制：

```yaml
ui:
  roots: ["."]                # 允许的起点，"." 是项目根
  allow_system_drives: true   # 默认可从盘符开始浏览整台机器
  max_entries: 400
```

服务默认只监听 `127.0.0.1`，所以对本机工具放开整台机器是合理取舍。
**若要用 `serve --host 0.0.0.0` 暴露到局域网，务必把 `allow_system_drives` 改成 `false`**，
只留 `roots` 里明确列出的目录；越界访问返回 403。产物文件名另走白名单正则
（`original|canonical|overlay|boxes|crop\d+\.png` + 16 位十六进制 token）。

## 已知边界

- **视角必须分流**。不同面的图混在一个视角里，每个面都会被其他面判成缺陷。
- **带标注的截图不能进数据集**。红色框会被当成产品特征；`preprocess.annotation.reject_area_ratio` 会自动拒绝这类图。
- 良品少于 `min_good_per_view`（默认 30）时 `fit` 会告警——模型能跑但阈值不可信。
- STEFANN 的协方差按位置估计，良品数必须 > `pca_dim + 2`，否则退化。
- 只训良品意味着系统只会说「这里不像良品」，缺陷类型靠 `classify.rules` 的形状规则推断，规则需要针对实际产品调。

## 测试

```bash
$ADKIT_PY -m pytest tests -q
```

`tests/` 用合成图（矩形元件 + 人工裂纹/缺角）跑通全链路，不下载预训练权重。
