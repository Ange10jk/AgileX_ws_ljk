# RoboDojo Piper HTTP API 接入计划

GPU 机提供模型推理和 Piper IK，robot 负责采集观测、发起 HTTP 请求和控制双侧从臂。部署配置从 [base_RoboDojo.yaml](../configs/base_RoboDojo.yaml) 派生，选择 `[left, right]`、仅启用两个 follower，使用 deploy 模式，`record.backend: none`，手动启动策略。`config.py` 增加 `lcgen_http` 后端和连接配置。

| 环节 | 复用与改动 |
| --- | --- |
| 采集观测 | 复用 [相机快照](../piper_control/hardware/cameras.py)、[从臂快照](../piper_control/hardware/robot.py)和 [make_observation](../piper_control/observation.py)。每个动作步采集时间接近且有效的三路 RGB 与从臂实测状态，保留相机与反馈时间戳。按 API 所用 Piper URDF 对两侧实测关节角做 FK。 |
| 请求 API | 复用 [PolicyClient](../piper_control/policies/client.py) 的异步子进程与 `generation/seq`，在 [worker.py](../piper_control/policies/worker.py) 分派至新建的 `piper_control/policies/lcgen_http.py`。用显式禁用代理的 HTTP 客户端直连 `http://100.113.96.55:8089`。启动检查 `GET /health` 的 `ready`、`piper`、`joint`、14 维及 `runtime_fps=30`，推理调用 `POST /predict`。 |
| 解析动作 | 在新建的 **`piper_control/policies/lcgen_http.py` 响应解析器**中校验整段 `actions`，按双臂顺序组装并完成夹爪单位换算。向控制侧返回形状为 `(N, 14)` 的绝对目标数组，按实际返回的 `N = len(actions)` 执行，默认 8 步。 |
| 下发从臂 | 复用 [Session](../piper_control/workflows/session.py) 和 [Controller](../piper_control/control/arbiter.py) 的动作队列、模式切换和旧响应拒收逻辑。每个 policy tick 取一个目标，交给 [SDKRobot.command](../piper_control/hardware/robot.py) 按侧调用 `move_js()` 和 `move_gripper_m()`。 |

## 请求与执行时序

- `observations` 是按 `wam_frame_index` 从 0 连续排列的同一 episode 历史。每项的 `images` 含三路 JPEG/PNG 字节的 Base64（无 data URI 前缀），键名映射为 `cam_h → cam_high`、`cam_l → cam_left_wrist`、`cam_r → cam_right_wrist`，并带 `env_idx=0`、`wam_fps=30.0`。
- 最新项带非空 `instruction` 和完整 `state`。左右分别提供 `{side}_arm_joint_state`（按 `joint1…joint6` 排列的实测六元素数值数组，rad）、`{side}_ee_pose`（各自 `base_link → link6`，`[x,y,z,qw,qx,qy,qz]`，位置为米）和 `{side}_ee_joint_state`（单元素列表，实测夹爪宽度以米除以 `0.1`，填入有效的归一化值）。
- 设置 `observation.fps: 30` 和 `workflow.policy_hz: 30`。在 `Session` 用单调时钟按约 33.3 ms 间隔逐条提交目标，并核对实际下发间隔。第 0 帧在首条动作前采集。每下发一条模型目标，历史步号加一，并在下一拍采集对应观测。首段 8 步执行后，用第 8 步后的新观测发 `0..8`。等待 HTTP 时保持姿态，不推进步号。开始新 episode 时清除历史。发送前检查请求体的 16 MiB 上限。

## 关节顺序与夹爪换算

每步从 `left_arm_joint_state` 和 `right_arm_joint_state` 的六元素数值数组依次读取 `joint1…joint6`，按 `[左 joint1…joint6, 左夹爪, 右 joint1…joint6, 右夹爪]` 排列。API 的夹爪字段 `left_ee_joint_state` 和 `right_ee_joint_state` 各有一个值，范围 `[0, 1]`，0 为闭合，1 为 0.1 m 开口。在 **HTTP 响应解析器**里，校验后只做以下两处换算，每行执行一次，位置为零基索引。其余 12 个关节角按 rad 原样传递。

```python
action[6] *= 0.1
action[13] *= 0.1
```

## 超时与非法动作

- 配置 HTTP 连接与请求超时，并让策略子进程的 watchdog 覆盖请求期限。请求失败、超时、HTTP 422 或响应过期时，记录状态码、耗时及可用的 `request_id`，清空动作队列并切到 HOLD。递增 `generation` 丢弃迟到的响应。子进程卡住时回收重启。操作员确认后以新 episode 取实测状态再开始。
- 整段入队前校验两侧各 6 个有限关节角和一个有限夹爪值、夹爪归一化范围、实机关节限位，以及首步相对最新实测关节和后续相邻目标的变化量。任一步不合法就拒收整段，走同一清队列和 HOLD 流程。CAN 下发失败或反馈不可用时走现有控制器故障停止路径。
- 验收先用 mock 或录制观测验证左右顺序、夹爪端点 `0 → 0 m` 和 `1 → 0.1 m`、30 Hz 下发时序，以及超时或非法动作时整段无下发。再从 robot 做只请求不下发的实拍联调，核对输出和延迟后进行实机测试。
