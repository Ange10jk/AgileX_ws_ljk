# 第一阶段：Piper 双臂 RoboDojo 真实任务数据采集计划

范围：两对 Piper 主从臂（left/right），顶部 `cam_h`、左腕 `cam_l`、右腕 `cam_r` 三台 RealSense；只采集 RoboDojo **Piper** 本体的 6 个真实桌面任务。此阶段的产物是逐条 ROS bag；不涉及策略部署、评测或 bag 到 LeRobot 的转换。以下按本仓库当前代码和配置编写。

## 1. 先确认要改什么

| 文件 | 本阶段处理 | 原因 |
| --- | --- | --- |
| `configs/collect_dual.yaml` | **必改**：把 `observation.task` 写成当前任务的准确指令；建议为每个任务各建一份继承此文件的任务 YAML，避免每次覆盖公共配置。 | `collect_dual.yaml` 已继承 `collect.yaml`，选中左右双臂、30 Hz 控制、15 Hz 主从动作、`auto_start: true`、`max_episode_sec: null`、`record.backend: ros_publish`，这些不用改。任务指令当前继承自 `base.yaml` 的插头任务，不能直接用于全部 6 个任务。 |
| `configs/base.yaml` | **必须核对，值不符才改**：三台相机的 `serial` 与实际 `cam_h/cam_l/cam_r` 一一对应；分辨率、帧率（现为 640×480、15 fps）和物理安装视角符合方案。 | `camera_map` 已有三视角。`record.output_dir` 对 `ros_publish` 控制进程不生成 bag；录包位置由 `record.py --output` 决定。不要为任务反复修改全局 `observation.task`。 |
| `configs/hardware/can.yaml` | **必须核对，值不符才改**：四个 USB-CAN 序列号严格对应 `left_leader`、`left_follower`、`right_leader`、`right_follower`。 | 配置按 USB 序列号解析，不依赖变化的 `canN` 编号；不可把主从或左右映射反。 |
| `configs/collect.yaml` | 通常无需改。 | 已开启两侧 leader、`observation.depth: true`、`align_depth: true`、`workflow.mode: collect`，并指定 `recording/topics.yaml`。其 `record.backend: ros` 会被 `collect_dual.yaml` 的 `ros_publish` 覆盖；不要直接用 `collect.yaml` 替代逐 episode 录包方案。 |
| `configs/recording/topics.yaml` | 通常无需改，只核对展开清单。 | 已要求双臂状态/命令/动作以及三相机 RGB、原始深度、对齐深度和内参。关闭深度却保留必需 topic 会导致不完整。 |
| `scripts/prepare_can.py`、`scripts/run.py`、`scripts/record.py` | **无需改代码**。 | 分别是 CAN 准备、控制流程和独立 rosbag 录制入口；现有参数足够完成第一阶段。 |

推荐新增 6 份任务配置（实际采集前创建；本计划本身不修改它们）：`configs/tasks/collect_<task>.yaml`，内容最少为：

```yaml
extends: ../collect_dual.yaml
observation:
  task: <该任务实际给操作者和后续数据集使用的完整指令>
```

`extends` 按 YAML 所在目录解析。每次改配置后重启控制进程，不要在正在录制时改文件。配置加载器会合并继承配置；`camera_map` 是替换式字段，任务 YAML 不必重复写它。保留 `record.backend: ros_publish`，否则可能变成自动录包模式。

六个 Piper 任务目录名固定为：`stack_and_cover_blocks`、`fill_pen_holder`、`put_objects_into_basket`、`insert_charger`、`stack_bowls`、`stand_up_bottles`。不要把 Piper X 或 ARX X5 的任务混进来。具体物件摆放、复位图、自然语言指令和完成标准，逐项以 [RoboDojo 真实任务说明](https://robodojo-benchmark.com/doc/real-tasks/)为准；采集前把选定指令填入该任务 YAML 并固定版本。

## 2. 数据目录与记录约定

建议把袋文件放到容量充足的绝对路径，例如 `/data/piper_robodojo_real/<task>/episode_0001/`。每条 episode 一个**全新**目录，录制脚本拒绝覆盖已有 `data.bag`、`data.bag.active`、`manifest.json` 或 `rosbag.log`。每个目录正常完成后包含上述前三者（不应残留 `.active`）。另在仓库外维护采集台账，逐条记任务、episode 编号、配置版本或 Git commit、场景/物体复位编号、操作者、日期、成败与异常；`manifest.json` 的 `complete` 只检查话题存在，不代表任务成功。

示例：

```text
/data/piper_robodojo_real/
  stack_and_cover_blocks/episode_0001/{data.bag,manifest.json,rosbag.log}
  fill_pen_holder/episode_0001/{data.bag,manifest.json,rosbag.log}
  put_objects_into_basket/episode_0001/{data.bag,manifest.json,rosbag.log}
  insert_charger/episode_0001/{data.bag,manifest.json,rosbag.log}
  stack_bowls/episode_0001/{data.bag,manifest.json,rosbag.log}
  stand_up_bottles/episode_0001/{data.bag,manifest.json,rosbag.log}
```

## 3. 每次开机与各终端命令

以下命令是**未来实机采集时由操作员执行**，不是本计划编写时执行。先完成设备断电/机械安装检查、急停可触达检查和桌面清场；确认 ROS Noetic、控制环境、磁盘容量和三相机连接。所有终端使用同一个 `ROS_MASTER_URI`，在仓库根目录执行；命令中的任务名、路径按当前任务替换。

**终端 A：ROS master**（已有可用 master 时复用）：

```bash
cd /home/ljk/workspace/AgileX_ws
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://localhost:11311
/usr/bin/python3 /opt/ros/noetic/bin/roscore
```

**终端 B：控制环境、CAN 与双臂主从控制**：

```bash
cd /home/ljk/workspace/AgileX_ws
export ROS_MASTER_URI=http://localhost:11311
PIPER_CONFIG=configs/tasks/collect_stack_and_cover_blocks.yaml
python scripts/prepare_can.py --list
python scripts/prepare_can.py --config "$PIPER_CONFIG"   # 只预览解析结果和 ip link 命令
python scripts/check_config.py --config "$PIPER_CONFIG"
sudo -v
python scripts/prepare_can.py --config "$PIPER_CONFIG" --execute
python scripts/run.py --config "$PIPER_CONFIG" --execute
```

`--execute` 会实际配置 CAN、连接并使能机械臂。先核对四个序列号到物理臂的映射以及相机画面，再执行。启动时程序可能使主臂向从臂当前姿态移动；确认工作空间无人手、无障碍。此配置会自动进入 `HUMAN`，主从控制进程持续发布 ROS 话题，但**不自动创建 bag**。控制终端看到人工 action 已就绪、主从跟随和三视角正常后，才进入录包。

**终端 C：逐条 episode 录包**（同一任务配置）：

```bash
cd /home/ljk/workspace/AgileX_ws
export ROS_MASTER_URI=http://localhost:11311
PIPER_CONFIG=configs/tasks/collect_stack_and_cover_blocks.yaml
python scripts/record.py --config "$PIPER_CONFIG" --list-topics
python scripts/record.py --config "$PIPER_CONFIG" --output /data/piper_robodojo_real/stack_and_cover_blocks/episode_0001
```

`--list-topics` 只展开清单，不连接机械臂。录制终端显示启动后，先确认 `rostopic list` 中有预期 `/piper/...` 话题，再开始示教。每段结束在**终端 C 按 Ctrl+C**，等待脚本打印 `Saved: .../data.bag` 且退出码为 0，再检查清单；不要直接杀死 rosbag 或关闭终端。下一条用 `episode_0002` 等新目录重新运行录包命令，终端 B 的主从进程可持续运行。更换任务时，先在控制终端输入 `q` 回车退出，再切换任务 YAML、复位场景并重启控制进程。

## 4. 一条 episode 的操作顺序

1. 根据官方任务说明摆好物体，确认相机视野、光照与桌面边界；记录复位编号。启动主从控制并等待左右跟随与相机正常。
2. 启动该 episode 的独立录包，确认 `manifest.json` 进入 `recording`，等待几秒确认图像/动作流已出现，再进行示教。录制应覆盖初始静止画面、全部动作和终态。
3. 在控制终端完成任务后输入 `s`（成功）或 `f`（失败）并回车；中途放弃用 `t`（截断）。这些会发出 `episode_end` 事件并让控制器暂停。`hold` 只暂停，不代表 episode 结束；`q` 会退出控制会话并把活动 episode 记为截断。若希望下一条仍由同一控制进程采集，复位后输入 `h` 回车重新进入人工控制。由于 `auto_start: true` 只在会话启动时生效，后续 episode 不会自动开始。
4. 在录制终端 Ctrl+C，等待 bag 索引关闭并验收。**一个 bag 对应一条 episode**；在下一条录包开始前完成场景复位，避免把跨任务复位动作混入示教。记录成功/失败和异常原因；失败示教可保留但要标注。

## 5. 完整性验收

每条录制结束后检查：

```bash
python -m json.tool /data/piper_robodojo_real/stack_and_cover_blocks/episode_0001/manifest.json | less
source /opt/ros/noetic/setup.bash
/usr/bin/python3 /opt/ros/noetic/bin/rosbag info /data/piper_robodojo_real/stack_and_cover_blocks/episode_0001/data.bag
```

验收门槛：`manifest.json` 的 `status` 为 `complete`，`missing_required_topics` 为空；`data.bag` 非空且可由 `rosbag info` 读取，无 `.bag.active`；`rosbag.log` 无 `buffer exceeded` 或 `dropping oldest`。当前 `topics.yaml` 在双臂三相机下要求 `/piper/state`、`/piper/action`、`/piper/action_metadata`，左右各自的从臂 `joint_states`/`diagnostics` 和主臂 `command`，以及每台相机的 color、depth、aligned_depth_to_color 的 `image_raw` 与 `camera_info`。主臂 `joint_states`/`diagnostics` 与 `/piper/events` 是可选的；主臂 `command` 是控制目标而非实测关节状态。还应抽看三视角 RGB/深度画面、检查时长与动作连续性、确认该 bag 仅对应一条示教，并与台账中的实际结果一致。`complete` 只保证每个必需话题至少有一条消息，不能证明帧率稳定、时间同步、无丢帧或成功完成任务；必要时再做时间戳/频率和人工画面复核。

## 6. 安全与排障

- 操作者和旁观者避开双臂扫掠区；保持急停可触达。`stop` 是控制终端的紧急停止命令，紧急情况优先使用硬件急停。安全恢复前不要重新使能；可按仓库 `scripts/recover.py` 对应流程处理，不要盲目重复启动。
- CAN 解析失败：先用 `prepare_can.py --list` 对照四个 USB 序列号、线缆和左右主从物理连接；无 `--execute` 的配置预览应显示四个正确的 `canN`。不要仅凭 `canN` 编号判断身份。
- 相机缺失或画面错位：核对 `base.yaml` 中三个唯一序列号和 `camera_map`，确认实际安装与视角。缺少深度或对齐深度时检查 `collect.yaml` 的两个开关、RealSense 带宽和连接；不要通过删掉必需 topic 掩盖缺流。
- bag 缺话题：确认终端 B/C 用同一任务配置和同一 `ROS_MASTER_URI`，控制进程仍在发布；先用 `--list-topics` 看展开结果，再检查 `manifest.json` 的 `missing_required_topics` 与 `rosbag.log`。`ros_publish` 下未启动 `record.py` 时没有 bag 是预期行为。
- `incomplete`/`failed`、`.active` 残留、缓冲溢出或时间戳异常的 episode 不作为合格示教；保留原始文件以排障，用新编号重录，勿覆盖或手动把 manifest 改成 `complete`。录制前预估 RGB 加双深度的磁盘吞吐，确保空间足够。

参考：本仓库 `README.md`、上述 YAML、`piper_control/cli.py`、`piper_control/workflows/session.py`、`piper_control/recording/manual_bag.py`、`piper_control/recording/topics.py`；任务名称参考 [RoboDojo 真实任务目录](https://robodojo-benchmark.com/doc/real-tasks/) 与 [官方项目](https://github.com/RoboDojo-Benchmark/RoboDojo)。
