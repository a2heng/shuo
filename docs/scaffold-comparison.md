# Tauri vs Electron — 脚手架对比（说 · Shuo）

> 目的：**同一套推理、同一套前端**，只换外壳，比较两个脚手架在「输入」和「上屏」上的真实差异。
>
> 实测环境：Ubuntu 26.04 / GNOME 50 Wayland / Python 3.14 / Node 22 / Electron 40.10.6 / Tauri 3.0.0-alpha.5。
> 推理引擎：`src-tauri` 编译出的 `shuo --engine`（Electron 通过 stdin/stdout JSON Lines 复用）。

## 结论（先行）

**保留 Tauri，去掉 Electron 实现。**

原因不在于「谁更流行」，而是这个应用的三个核心动作——**按住说话、识别、上屏**——Tauri 都能在进程内/系统层完成；Electron 每一条都要额外借一个原生进程或原生模块，最后等于「Electron 壳 + Rust 引擎」，反而失去了 Electron 的意义。

| 维度 | Tauri | Electron |
|---|---|---|
| 按住说话（按下/松开） | ✅ 原生低级钩子 / 门户信号 | ❌ `globalShortcut` 只有按下 → 只能做「按一下开始、再按一下结束」 |
| 推理 | ✅ 进程内直接调用 | ⚠️ 子进程 + JSON/Base64（本实现）或另写 native addon |
| 系统音频 | ✅ cpal 直接拿设备名/采样率 | ⚠️ Chromium 音频栈（能跑，但设备标签、权限、MediaRecorder/AudioWorklet 都有坑） |
| 上屏（Windows） | ✅ SendInput + `KEYEVENTF_UNICODE`（与 Python 版同一套） | ❌ 无内置键盘注入；要 native 模块或借 Rust 引擎 |
| 上屏（Linux） | ⚠️ 平台受限（见下），但代码在应用内 | ⚠️ 同样受限，且自身没有注入能力 |
| 包体 | ✅ deb 16 MB / AppImage 96 MB | ❌ Electron dist 289 MB（开发态） |
| 常驻内存（模型已加载） | 3.27 GB | 3.59 GB（引擎 3.12 GB + Chromium ≈0.47 GB） |
| 启动到 ASR 就绪 | ≈3 s（模型加载占绝大部分） | ≈3–4 s（多一层 Chromium 启动） |
| 托盘 | ✅ 可用 | ✅ 更成熟（Linux 上略稳） |
| 渲染一致性 | ⚠️ 依赖系统 WebKitGTK | ✅ 自带 Chromium，跨发行版一致 |

## 一、输入直观性（谁的操作更少）

| 操作 | Tauri | Electron |
|---|---|---|
| 开始录音 | 按住 F2（1 个动作） | 按一下 F2（第 1 次） |
| 结束录音 | 松开 F2（同一次按键的抬起） | 再按一下 F2（第 2 次） |
| 鼠标侧键 | ✅ 支持（XButton1/2） | ❌ `globalShortcut` 不支持鼠标键 |
| 误触/忘记停止 | 松开即停 | 容易一直录（要记得再按一次） |
| 界面按钮 | 按住说话（与 Python 版一致） | 按住说话（同一套前端，行为一致） |

**结论：Tauri 明显更直观。**「按住—说—松开」是语音输入最自然的一次连续手势；Electron 因为拿不到 KeyUp，只能做成切换式，多一次按键且容易忘记停止。

> 补充：Electron 在 Wayland 下要加 `GlobalShortcutsPortal` 特性才能注册全局键；即使这样，门户/`globalShortcut` 也只有激活（按下）通知，没有松开。

## 二、上屏（谁做得更好）

### Windows

- **Tauri**：`SendInput` + `KEYEVENTF_UNICODE`（照搬 Python 版 `send_unicode_text`），不经过剪贴板，中文/emoji 都能打，终端/Vim 也正常。
- **Electron**：没有内置键盘注入 API。要么上 `nut.js`/`robotjs` 这类 native 模块（要编译、CJK 支持不一），要么像本实现一样把活交给 Rust 引擎。

→ **Tauri 赢**，而且这是「Python 版原有能力」的直接延续。

### Linux（这是平台问题，不是脚手架问题）

实测结论（GNOME 50 / mutter）：

1. **逐字符 keysym 注入不可用**：RemoteDesktop 门户的 `NotifyKeyboardKeysym` 会把按键交给**当前输入法**。自检里注入 `abc123`，文本框里出现的是「阿拔出23」——libpinyin 把 a/b/c 当拼音转成了汉字。中文更会被候选词吃掉。
2. **剪贴板 + 粘贴快捷键也不够**：Ctrl+V / Shift+Insert 在 GUI 里能用，但 **TUI / 终端里粘贴不可靠**，且会覆盖用户剪贴板（可延迟还原，仍有竞态）。
3. **IBus 输入法引擎**（`commit_text`）是理论上最正确的路：文本不经过模拟按键，TUI 也能上屏。但代价是：
   - IBus **不支持用户目录组件**（源码里用户目录那段是 `#if 0`），必须装到 `/usr/share/ibus/component`（要 root）；
   - 需要把 `shuo-voice` 加进 GNOME 输入源，并在每次上屏前后切换输入法（或不切、让用户手动切）；
   - 普通打字会失去拼音（引擎只能透传），所以只能「临时切过去再切回」。
   - 本机实测到「引擎能被 daemon 启动、socket 能连上」，但 `SetGlobalEngine` 仍报 `Factory` 对象不存在；再叠加切换成本，**判定为不够优雅，暂不采用**（相关试验代码已从仓库移除，避免误导）。

所以 Linux 上目前的做法是：**剪贴板 + 粘贴**为主，逐字符注入仅作最后手段。最终做成**两个独立热键**（各自带开关），而不是设置里切换：

- 主热键（默认 `F2`）→ `Ctrl+V` 上屏（图形界面，GTK 实测可用）；
- 终端热键（默认 `F3`）→ `Ctrl+Shift+V` 上屏（终端 / TUI 的粘贴键）。

实测没有「一个键通吃」：`Ctrl+Shift+V` 在 GTK TextView 里无反应，`Ctrl+V` 在终端里只是 quoted-insert。用哪个热键说话就用哪种上屏，少一次设置切换。

两个脚手架在这里没有胜负——**都需要额外的系统集成**，Electron 并不会更好。

> 如果以后要做「真正优雅的 Linux 上屏」，方向是：把 IBus 引擎作为独立小组件安装（deb 直接放系统目录）、默认不切换输入法，只在检测到「TUI/无粘贴能力」时提示用户切到 Shuo Voice。这是一条独立的产品线，和 Tauri/Electron 之争无关。

## 三、其余实测数据

| 指标 | Tauri | Electron |
|---|---|---|
| 安装包 | `Shuo_<版本>_amd64.deb` **16 MB**；AppImage **96 MB** | Electron dist **289 MB**（未压缩）；打包后通常 120–200 MB |
| 进程数（空闲） | 1（+ ORT 线程） | 4–6（main / renderer / GPU / utility）+ 引擎子进程 |
| RSS（模型已加载） | **3.27 GB** | **3.59 GB** |
| 推理传输 | 无（同进程） | 每 10 s 音频 ≈ 0.9 MB Base64（48 kHz f32） |
| 音频设备名 | cpal 直接给 ALSA/PipeWire 名字 | Chromium 标签（与系统名不完全一致） |
| 音频采集实现 | cpal 常开 + 严格热键边界（无预缓冲/收尾等待） | `getUserMedia` + AudioWorklet（本实现；MediaRecorder 的 webm 解码时长不可靠） |
| 自动输入 | 引擎内 `type_text`（SendInput / 门户） | 需借引擎或 native 模块 |
| 托盘 | 可用 | 更成熟 |
| 升级/分发 | 依赖 WebKitGTK（deb 声明依赖） | 自带 Chromium，发行版无依赖差异 |

> 内存的大头是模型和 ONNX Runtime（3.1 GB），脚手架本身的差距约 **0.3 GB**；也就是说单看内存，Electron 不是决定性劣势，但叠加「输入 + 上屏 + 包体」后就不划算了。

## 四、最终取舍

- **保留**：`src-tauri/`（Tauri 3）——输入最直观、上屏有原生实现、包体小、推理零拷贝。
- **去掉**：`electron/` 对照实现——它的存在已经完成了对比使命；保留会让仓库出现两套需要同步维护的外壳，而 Electron 版的核心能力仍然要靠 Rust 引擎兜底。
- **保留本文档**：记录实测结论（尤其是 Linux 上屏的三个坑），避免以后重复踩。

如果将来有「必须用 Chromium 统一渲染」或「团队只写 JS」的硬需求，可以按本文的记录快速复活一个 Electron 外壳（对照实现已删除，避免双外壳长期维护）；但那就不是这个项目的默认形态了。
