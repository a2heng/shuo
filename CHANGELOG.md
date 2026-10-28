# Changelog

## 未发布 — 提示词控制输出风格的 ASR 应用（llama.cpp/GGUF · Qwen3-ASR-1.7B）

### 新增 — 输出风格：提示词控制转写文本风格
- 微调 Qwen3-ASR-1.7B 为「微调模型包」：用户提示词进 chat 模板 system 段（context 通道），模型直接输出目标风格文本；风格预设：逐字转写 / 中文书面化 / 英文书面化 / 中文（翻译）/ 英文（翻译）/ 简略（压缩要点）/ 详细（保留全部细节）/ 自定义提示词
- 基座包不执行指令（官方防指令注入设计），微调模型包（目录带 `shuo-model.json` 标记：基座 / 适配器 / 提示词集记录）才兑现风格开关；基座上运行不报错，提示词退化为识别偏置
- 本仓库是「调用模型的应用程序」：微调产线不在仓库内，模型包以 GitHub Release 资产分发
- **模型一键下载**（`src/model_download.rs`）：缺模型时横幅与设置页一键从 Release 拉取微调模型包（约 1.4 GB）到 `~/.shuo/models/Qwen3-ASR-1.7B-GGUF-tuned/`；断点续传（.part + Range）、大小校验、400ms 节流进度事件、完成后自动重载引擎；TLS 证书走**系统信任库**（ureq `platform-verifier`），代理环境的拦截证书装进系统即可下载；下载源 = 文件顶部 `RELEASE_REPO`/`RELEASE_TAG` 两个常量（当前指向贡献者 fork 的 pre-release，上游换成自己的 Release 即可）

### 变更 — ASR 推理框架迁移到 llama.cpp/GGUF（换用 Qwen3-ASR-1.7B）
- 新后端 `src/gguf_asr.rs`：llama-cpp-sys-2（=0.1.157，openmp+mtmd）**源码静态链接**进可执行文件（无 DLL/so 伴生），mtmd 音频 tokenize/prefill + 贪心解码 + `language X<asr_text>` 解析；与 ONNX 路径同接口（AsrResult），长音频复用静音切分
- 双后端调度：`asr_backend` 配置（auto/onnx/gguf；auto = 有 GGUF 模型优先 GGUF）；所选后端加载失败自动回退另一个（日志说明）；`Status.asr_backend` 上报，UI chip 显示当前后端
- ort 继续承担降噪模型（resources/denoise.onnx）与 ONNX ASR 后备后端（Daumee 0.6B 包）
- 打包：deb depends 增加 `libgomp1`（OpenMP）；ci/release 工作流补 libclang 准备（bindgen）；Windows 运行期依赖 VC++ 2015-2022 运行库（vcomp140）

### 移除 — 双模型工作流 / PySide legacy 前端 / 微调产线
- **独立 LLM 后处理段（Qwen3.5-2B ONNX）整体移除**：`src/llm.rs`（583 行）及全部关联（`mod llm`、加载器、worker LLM 段、5 个命令、`half` 依赖、LLM 模型槽位、UI 设置组/chip/提示词库）；「逐字转写」预设即旧的不优化行为；旧配置字段名经 serde alias 承接、废弃字段经 `extra` flatten 原样保留，history.json 的 `llm_text` 仅兼容展示
- **结果区双卡合一**：「识别原文 / 整理结果」合并为单张「识别结果」卡（语言徽章 + 生效风格徽章 + 复制按钮）——单模型路径下 ASR 输出即最终文本，双卡恒同文
- **`legacy/2026-10-25-python/`（PySide6 原版）与 `i18n/`（Python 翻译工具）移除**：应用单一实现（Tauri 3），UI 多语言源 = `ui/locales/*.json`（27 语言，新键先进 app.js 的 EXTRA zh/en）
- **`finetune/` 微调产线与 `finetune.yml` 工作流移出仓库**：产线代码与全部实测记录保留在贡献分支 `feat/gguf-1.7b` 历史中，模型维护者可取用
- CI：`py-ui` job 精简为纯 `ui` job（node 语法 + locales JSON 解析 + 控件存在性 + **防回归断言**：LLM 控件/命令/模块、finetune/legacy/i18n 目录、旧话术与双卡片必须缺席）；rust 双腿新增 **测试构建 artifact**（`test-build-windows` / `test-build-linux`：`cargo build --release` 直出 + denoise.onnx + 使用说明，保留 14 天），实机测试不必等 release.yml 全量打包

## 26.10.25 — 2026-10-25

> 版本号自本版起改为日期 `YY.MM.DD`，tag 为 `v26.10.25`；同日多次发版自动加 `-2`、`-3`。

### 新增 — Tauri 3 全 Rust 重写（`src-tauri/`）
- **推理全部搬到 Rust**：`ort` 加载 Qwen3-ASR-0.6B（mel → conv → transformer → decoder init/step）与 Qwen3.5-2B（embed + decoder + 混合线性注意力状态），不再需要 Python / PyTorch / onnxruntime-python
- **数值对齐**：mel 滤波器与 librosa 一致（max diff 3.7e-9），mel 谱 2.1e-5，encoder 8.7e-7；LLM chat prompt 与 `chat_template.jinja`（`enable_thinking=false`）逐 token 对齐（`llm::tests::chat_prompt_token_ids_match_python`）
- **原生音频**：cpal 常开采集 + 0.5 s 预缓冲 + 500 ms 收尾，AGC 归一化到 -1 dBFS
- **历史记录**：默认开启，文件与内存都限制 500 条；关闭时界面明确提示
- **界面**：小录音按钮 + 顶部结果卡片 + 下方历史占主要空间；历史时间单独一行，避免与正文重叠
- **状态胶囊**：底部（任务栏上方）显示「录音中 + 不规则波形」/「识别中 + 转圈」；XWayland 小窗置顶、不抢焦点、鼠标穿透，不需要 GNOME 扩展或注销
- **Linux 上屏改用 uinput 虚拟键盘**：不再每次启动弹 RemoteDesktop「共享」授权；uinput 不可用时退回门户（restore token 记住授权，被撤回会自动重新申请）
- **前端**：纯静态 HTML/CSS/JS（`src-tauri/ui/`），复用 27 种语言包；设置抽屉、历史、结果卡片、托盘
- **Windows 热键**：低级键鼠钩子（`WH_KEYBOARD_LL` / `WH_MOUSE_LL`），支持按住说话与鼠标侧键，过滤自身注入事件
- **Linux 热键**：优先 xdg-desktop-portal `GlobalShortcuts`（GNOME/KDE Wayland，按下+松开信号），portal 不可用时 evdev 直读 `/dev/input/event*`；deb 自带 udev 规则
- **两个独立热键**（各带开关）：主热键（默认 F2）→ Ctrl+V 上屏（图形界面）；终端热键（默认 F3）→ Ctrl+Shift+V 上屏（终端 / TUI）
- **Linux 上屏**：剪贴板 + 粘贴快捷键（门户/工具发键）；逐字符 keysym 因会被当前输入法改写而不再使用
- **Windows 上屏**：`SendInput` + `KEYEVENTF_UNICODE`（与 Python 版 `send_unicode_text` 同一套）
- **打包**：Linux deb（16 MB，含降噪模型、udev 规则、门户桌面文件）/ AppImage（96 MB）；Windows NSIS 配置 + `build.ps1`
- **门户集成**：启动时注册 host app id（`com.shuo.desktop`），并确保 `com.shuo.desktop.desktop` 存在（AppImage 也能用）

### 变更
- 与 Python 版共用 `~/.shuo/`（config/history/log/custom_prompts），配置字段兼容
- 模型不随安装包分发：配置路径 → `SHUO_MODEL_DIR` → `~/.shuo/models` → 可执行文件上溯 → 仓库目录

### 记录
- 新增 [Tauri vs Electron 脚手架对比](docs/scaffold-comparison.md)：同一套推理/前端下比较输入、上屏、包体、内存；结论保留 Tauri，Electron 对照实现已删除

## 0.1.11 — 2026-06-27


### 新增
- **麦克风选择下拉框**：工具栏可选择录音设备，选择持久化到配置
- **预缓冲录音**：持续维护 0.3s 滚动缓冲，按下时回溯截取，解决 MME 启动慢丢前音问题
- **自动增益控制 (AGC)**：录音后 peak 归一化到 -1 dBFS
- **自动采样率适配**：按设备原生采样率录音，ASR pipeline 自动重采样到 16kHz
- **中文设备名修复**：修复 PortAudio 在中文 Windows 上返回 GBK 乱码设备名
- **设置菜单**：齿轮图标下拉菜单，集中管理所有设置项

### 优化
- **UI 精简**：去掉底部栏和 hint 文字，录音按钮移入 toolbar，窗口更紧凑
- **下拉框统一**：ASR 语言和麦克风下拉框统一 140px 宽度，带文字标签说明用途
- **设备列表只显示 MME**：移除 WASAPI（兼容性问题），去掉 API 前缀
- **i18n**：新增 `settings.mic` / `settings.mic_default` / `settings.asr_lang_label` / `settings.asr_lang_tip` / `settings.ui_lang` 到全部 28 个语言文件

## 0.1.10 — 2026-06-03

### 修复
- **首次启动语言显示错误**：配置为中文但界面显示英文，因 `i18n.load()` 在 UI 控件创建后才调用，已提前至 `__init__` 开头
- **托盘右键退出未随语言切换刷新**：`quit_action` 未存为成员，`_switch_lang` 未更新其文字

## 0.1.9 — 2026-06-03

### 新增
- **27 语言全覆盖**：UI 界面 + ASR 识别语言下拉框全部国际化，每个语言一个 locale 文件，默认英文界面
- **ASR 识别语言选择**：工具栏下拉框，支持 27 种语言 + 自动检测，传入模型作为语言参数
- **识别排队**：识别中再录音自动排队，当前识别完成后自动处理下一条
- **中文专属去标点**：中文界面下可开启"去标点替换为空格"，其他语言隐藏

### 修复
- **线程崩溃**：识别中再按侧键导致 QThread 销毁时崩溃

### 优化
- **历史加载顺序**：最新记录显示在最上面
- **README 目录结构**：文件夹在前、文件在后，按字母序排列
- **i18n 回落机制**：英文兜底，新增语言不会显示 key 名

## 0.1.8 — 2026-06-03

### 修复
- **i18n 翻译不完整**：清理 locales 中 12 个未使用的过期 key；切换语言时遗漏刷新历史记录、清空、关于、退出等控件，已补全

### 优化
- **onnx_inference.py 移至根目录**：不再放在 `Qwen3-ASR-0.6B-ONNX-CPU/` 下载目录内，避免被覆盖；模型目录完全 gitignore
- **根目录恢复 mel_filters.npy**：下载目录中的 npy 含 pickle 数据导致加载失败，改用根目录自带的纯净版本，无需 `allow_pickle`

## 0.1.7 — 2026-06-03

### 修复
- **内存泄漏**：录音 PCM buffer 即时清空、旧线程 `wait()` + `deleteLater()`、临时 WAV 用完即删、卡片上限 100 条自动淘汰
- **ONNX 内存**：关闭 `enable_cpu_mem_arena`，推理内存从数百 MB 降到几十 MB，不再随录音次数增长

## 0.1.6 — 2026-06-03

### 隐私
- 日志不再记录识别内容
- 历史记录默认关闭

## 0.1.5 — 2026-06-03

### 优化
- **砍掉 librosa 依赖**：音频加载改用 stdlib `wave`，STFT 改用 `numpy.fft`，mel 滤波器预计算为 `mel_filters.npy`
- **砍掉 scipy / soundfile / numba**：全部替换为 numpy + stdlib，只用 `onnxruntime + numpy + tokenizers` 三个外部依赖

### 新增
- 应用图标 `shuo.ico`

## 0.1.4 — 2026-06-03

### 修复
- 打包命令加回 `--hidden-import librosa`，修复运行时模块缺失

## 0.1.3 — 2026-06-03

### 优化
- 打包命令加 `--exclude-module torch/sklearn/numba/llvmlite/tensorflow`，产物从 7.5GB 降到 2.7GB
- 打包命令从多行改为单行，避免 PowerShell 续行符问题

## 0.1.2 — 2026-06-03

### 修复
- 打包命令写入 README，支持 Windows/macOS/Linux
- 补 `--hidden-import librosa`

## 0.1.1 — 2026-06-03

### 新增
- 语言切换改为按钮式：🌐 图标 + ZH/EN 文字，点击弹出菜单

### 修复
- 补充 `QGridLayout` 导入，修复关于对话框无法打开
- 语言下拉框翻译 key 未加载的问题

## 0.1.0 — 2026-06-03

### 新增
- 系统托盘图标，支持最小化到托盘
- 全局热键监听（键盘 + 鼠标侧键）
- 中/英文界面切换，完整 i18n 覆盖（47 个 key）
- 关于对话框，含依赖列表和许可证信息
- 识别结果保存到本地历史（最多 500 条）
- 自动输入模式：识别完成后自动粘贴到前台窗口

### 优化
- **主题系统重写**：显式颜色方案，切换瞬间响应，亮/暗模式完全跟随 Windows 系统设置
- **UI 全面美化**：圆形录音按钮（系统强调色）、卡片式结果项（左 accent 色条 + hover 复制按钮）、旋转弧线加载动画、原生 Win11 风格滚动条
- **启动加速**：去掉冗余初始化，模型后台加载，界面立即可见
- **窗口美化**：恢复最小化按钮、系统阴影、Win11 圆角，标题栏跟随主题色

### 修复
- 主题切换时背景色不跟随的问题
- 录音按钮 hover 颜色消失的问题
- 结果卡片 hover 时文字跳动的问题
- 浅色主题下复制图标不可见的问题
- 翻译文本未加载导致界面显示 key 名的问题
