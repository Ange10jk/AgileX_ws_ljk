# 第一阶段：Piper 双臂 RoboDojo 真实任务数据采集计划

范围：两对 Piper 主从臂（left/right），顶部 `cam_h`、左腕 `cam_l`、右腕 `cam_r` 三台 RealSense；只采集 RoboDojo **Piper** 本体的 6 个真实桌面任务。此阶段的产物是逐条 ROS bag；不涉及策略部署、评测或 bag 到 LeRobot 的转换。以下按本仓库当前代码和配置编写。

## 1. 先确认要改什么

推荐新增 6 份任务配置：`configs/RoboDojo/collect_<task>.yaml`，内容最少为：

```yaml
extends: ../collect_dual_RoboDojo.yaml
observation:
  task: <该任务实际给操作者和后续数据集使用的完整指令>
```

`extends` 按 YAML 所在目录解析。

六个 Piper 任务目录名固定为：`stack_and_cover_blocks`、`fill_pen_holder`、`put_objects_into_basket`、`insert_charger`、`stack_bowls`、`stand_up_bottles`。不要把 Piper X 或 ARX X5 的任务混进来。具体物件摆放、复位图、自然语言指令和完成标准，逐项以 [RoboDojo 真实任务说明](https://robodojo-benchmark.com/doc/real-tasks/)为准；采集前把选定指令填入该任务 YAML 并固定版本。

## 2. 数据目录与记录约定

建议把袋文件放到容量充足的相对路径。

## 3. 每次开机与各终端命令

**终端 A：ROS master**（已有可用 master 时复用）：

```bash
cd ~/luojk/AgileX_ws_ljk
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://localhost:11311
/usr/bin/python3 /opt/ros/noetic/bin/roscore
```

**终端 B：控制环境、CAN 与双臂主从控制**：

```bash
cd ~/luojk/AgileX_ws_ljk
conda activate piper-control
PIPER_CONFIG=configs/RoboDojo/collect_stand_up_bottles.yaml
export ROS_MASTER_URI=http://localhost:11311
sudo -v
python scripts/prepare_can.py --config "$PIPER_CONFIG" --execute
python scripts/run.py --config "$PIPER_CONFIG" --execute
```

`--execute` 会实际配置 CAN、连接并使能机械臂。先核对四个序列号到物理臂的映射以及相机画面，再执行。
遇到问题执行
```bash
python scripts/recover.py --config "$PIPER_CONFIG"
```

**终端 C：逐条 episode 录包**（同一任务配置）：

```bash
cd ~/luojk/AgileX_ws_ljk
conda activate piper-control
PIPER_CONFIG=configs/RoboDojo/collect_stand_up_bottles.yaml
export ROS_MASTER_URI=http://localhost:11311
python scripts/record.py --config "$PIPER_CONFIG" --list-topics
python scripts/record.py --config "$PIPER_CONFIG" --output ~/luojk/datasets/stand_up_bottles/episode_30fps_action30_new_log/
```

`--list-topics` 只展开清单，不连接机械臂。录制终端显示启动后，先确认 `rostopic list` 中有预期 `/piper/...` 话题，再开始示教。每段结束在**终端 C 按 Ctrl+C**，等待脚本打印 `Saved: .../data.bag` 且退出码为 0，再检查清单；不要直接杀死 rosbag 或关闭终端。下一条用 `episode_0002` 等新目录重新运行录包命令，终端 B 的主从进程可持续运行。更换任务时，先在控制终端输入 `q` 回车退出，再切换任务 YAML、复位场景并重启控制进程。

## 4. 一条 episode 的操作顺序

1. 根据官方任务说明摆好物体，确认相机视野、光照与桌面边界；记录复位编号。启动主从控制并等待左右跟随与相机正常。
2. 启动该 episode 的独立录包，确认 `manifest.json` 进入 `recording`，等待几秒确认图像/动作流已出现，再进行示教。录制应覆盖初始静止画面、全部动作和终态。
3. 在控制终端完成任务后输入 `s`（成功）或 `f`（失败）并回车；中途放弃用 `t`（截断）。这些会发出 `episode_end` 事件并让控制器暂停。`hold` 只暂停，不代表 episode 结束；`q` 会退出控制会话并把活动 episode 记为截断。若希望下一条仍由同一控制进程采集，复位后输入 `h` 回车重新进入人工控制。由于 `auto_start: true` 只在会话启动时生效，后续 episode 不会自动开始。
4. 在录制终端 Ctrl+C，等待 bag 索引关闭并验收。**一个 bag 对应一条 episode**；在下一条录包开始前完成场景复位，避免把跨任务复位动作混入示教。记录成功/失败和异常原因；失败示教可保留但要标注。


参考：本仓库 `README.md`、上述 YAML、`piper_control/cli.py`、`piper_control/workflows/session.py`、`piper_control/recording/manual_bag.py`、`piper_control/recording/topics.py`；任务名称参考 [RoboDojo 真实任务目录](https://robodojo-benchmark.com/doc/real-tasks/) 与 [官方项目](https://github.com/RoboDojo-Benchmark/RoboDojo)。
