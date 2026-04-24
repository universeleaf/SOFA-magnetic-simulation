# SOFA Magnetic Guidewire Simulation

这个项目基于 SOFA 搭建磁导航导丝仿真场景，目标是在血管模型中实现可运行的导丝推进、磁头转向、碰撞接触和路径跟踪。当前工程同时保留了 Python 场景逻辑和本地 C++ `ElasticRodGuidewire` 原生插件，便于在稳定性、速度和物理真实性之间做迭代。

## 主要内容

- `elasticrod` 原生导丝后端，支持磁力/磁矩驱动、接触约束和导丝调参与诊断。
- `beam` 兼容路径，便于在插件不可用时回退验证。
- 多条中心线路径与血管网格资源，支持切换入口/出口路线。
- 诊断脚本，可快速检查推进、磁场、接触和 GUI 墙钟性能。
- RL 训练接口，包含 `gymnasium` 环境和 `stable_baselines3` 的 PPO 训练脚本。

## 目录说明

- `main.py`: 场景启动入口。
- `scene.py`: SOFA 场景组装。
- `controller.py`: 导航、推进、磁控制和诊断逻辑。
- `config.py`: 主要参数入口，包括后端、路径、求解器、接触和磁场配置。
- `runtime.py`: SOFA / 插件发现与 `runSofa` 启动逻辑。
- `diagnose_elasticrod.py`: 轻量诊断与基准脚本。
- `native/ElasticRodGuidewire/`: 原生 C++ 插件源码。
- `assets/`: 血管网格、中心线、路由和运行时资源。
- `tools/prepare_vessel_4_0108.py`: 当前血管模型相关的资源准备脚本。
- `rl_env.py` / `train_rl.py`: 强化学习环境与训练入口。

## 环境要求

- Windows
- 已安装 SOFA，并能找到 `runSofa.exe`
- Python 环境需与 SOFA / SofaPython3 兼容
- 编译原生插件时需要:
  - Visual Studio Build Tools
  - CMake
  - 可用的 Eigen 头文件
  - 本地 SOFA CMake 包

可选依赖:

- `gymnasium`
- `stable_baselines3`
- `torch`

## 快速开始

### 1. 配置 SOFA 路径

优先推荐设置环境变量:

```powershell
$env:SOFA_ROOT="D:\SOFA_v25.06.00_Win64\SOFA_v25.06.00_Win64"
```

如果不设环境变量，`runtime.py` 会尝试使用里面写死的候选路径。

### 2. 编译原生插件

在项目根目录执行:

```powershell
.\build_plugin.bat
```

默认会在 `build/ElasticRodGuidewire` 下生成 DLL。若你想指定输出名，可以传一个参数:

```powershell
.\build_plugin.bat ElasticRodGuidewire
```

### 3. 启动场景

```powershell
python .\main.py --autoplay
```

或先静态打开:

```powershell
python .\main.py --no-autoplay
```

`main.py` 会自动去找 `runSofa.exe` 并以 `SofaPython3` 方式启动场景。

## 常用诊断命令

快速检查全链路:

```powershell
python .\diagnose_elasticrod.py --profile full --steps 120 --print-every 30
```

检查 GUI 墙钟推进性能:

```powershell
python .\diagnose_elasticrod.py --profile gui-benchmark --steps 80 --print-every 20
```

只看推进，不加磁场:

```powershell
python .\diagnose_elasticrod.py --profile push-only --steps 120 --print-every 20
```

## 强化学习

运行 PPO 训练:

```powershell
python .\train_rl.py --timesteps 200000 --sim-steps-per-action 5 --max-episode-steps 400
```

训练过程会使用:

- `ppo_guidewire_latest.zip`: 最近一次 checkpoint
- `ppo_guidewire_monitor.csv`: monitor 输出
- `ppo_guidewire_tensorboard/`: TensorBoard 日志

## 关键配置项

优先查看 `config.py` 里的这些入口:

- `GUIDEWIRE_BACKEND`: 选择 `elasticrod` 或 `beam`
- `ELASTICROD_STABILIZATION_MODE`: 选择 `strict` 或 `safe`
- `SELECTED_ROUTE_NAME`: 选择当前中心线路径
- `SCENE_AUTOPLAY`: 场景打开后是否自动播放

如果你在做推进/磁头/接触调试，通常先从 `config.py` 和 `controller.py` 两个文件入手。

## 相关说明文档

仓库里还保留了几份阶段性记录:

- `OPTIMIZATION_SUMMARY.md`
- `SAFE_MODE_OPTIMIZATION.md`
- `SAFE_MODE_PROPULSION_FIX.md`
- `guidewire_scene/ADVANCEMENT_FIX.md`

这些文档更偏实验记录和阶段结论，适合回看调参和问题定位过程。

## 常见问题

### 1. 找不到 `runSofa.exe`

- 先确认 `SOFA_ROOT` 是否设置正确。
- 再检查 `runtime.py` 中的候选路径是否和你机器上的安装路径一致。

### 2. 插件加载失败

- 先执行 `.\build_plugin.bat`。
- 再确认 `build/ElasticRodGuidewire` 下确实生成了 DLL。
- 如果你改过输出名，检查 `runtime.py` 里候选 DLL 名称是否覆盖到了你的产物。

### 3. 想切换血管路线

- 修改 `config.py` 中的 `SELECTED_ROUTE_NAME`。
- 路径文件索引在 `assets/centerline/extracted_paths/route_catalog.json`。

### 4. 想重做当前血管数据

- 参考 `tools/prepare_vessel_4_0108.py`
- 当前默认资源集中在 `assets/vessel_final_4_0108.*` 和 `assets/centerline/`

## 当前建议工作流

1. 先确认 `SOFA_ROOT`。
2. 编译 `ElasticRodGuidewire` 插件。
3. 用 `python .\diagnose_elasticrod.py --profile full ...` 做快速检查。
4. 再用 `python .\main.py --autoplay` 看 GUI 表现。
5. 稳定后再进入 RL 训练或更细的物理调试。
