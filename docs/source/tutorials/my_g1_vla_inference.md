# MY_G1 VLA inference with SONIC

`gear_sonic/scripts/run_my_g1_vla_inference.py` 是 Locomanip MY_G1 单 ego 相机模型的独立推理入口。它从 Isaac-GR00T PolicyServer 读取 checkpoint 的有效 modality 配置，向 SONIC C++ 发布 latent protocol v4 动作。

## 运行环境与启动

客户端 Python 环境需要 Isaac-GR00T、gear_sonic 的推理依赖以及 Pinocchio。服务器环境需要与 checkpoint 对应的 Isaac-GR00T processor 和模型实现。

### 一键启动本机推理

在 WBC 仓库根目录运行小型启动器，启动本机模型服务器、C++、MY_G1 客户端和键盘发布器：

```bash
python3 gear_sonic/scripts/my_launch.py \
    --model-path /path/to/checkpoint \
    --camera-host 192.168.123.164 \
    --prompt "pour water into the cup"
```

相机服务仍需在机器人上运行并提供 `ego_view`，设置见 [Data collection](data_collection.md)。启动器不启动相机或仿真。`--model-path` 必须指向可由当前 Isaac-GR00T 环境加载的本地 checkpoint 目录；启动器不修改 checkpoint 配置。

启动器本身只使用 Python 标准库和 tmux。服务器、客户端和键盘默认使用相邻 `Isaac-GR00T/.venv/bin/python`，无需 `.venv_inference` 或手动激活环境。可用 `--gr00t-root /path/to/Isaac-GR00T` 指定其他位置，该环境需同时具备客户端和服务器依赖。

默认 tmux 会话名为 `my_g1_inference`，`inference` 窗口布局如下：

| 位置 | 进程 |
| --- | --- |
| 左上 | PolicyServer，默认 `cuda:0`、本机端口 `5550` |
| 右上 | SONIC C++，默认 v1.1 模型，先加载 `scripts/setup_env.sh` |
| 左下 | MY_G1 客户端，等待服务器协议 ping 成功后启动 |
| 右下 | 键盘发布器，在这里输入键名后回车 |

启动后先在右上 C++ 窗格按回车确认 `deploy.sh`，等待 C++ 初始化和客户端就绪。随后在右下键盘窗格输入 `k` 回车、`i` 回车，确认初始姿态稳定，再输入 `p` 回车。启动器不自动发送这些控制键。输入 `t 新指令` 更新后续推理的 prompt。

客户端最多等待服务器 300 秒，每次 ping 超时 1 秒；超时或进程退出时保留窗格和输出，便于查看错误。客户端动作 horizon 自动从服务器读取。用 `Ctrl+b` 后按方向键切换窗格；`Ctrl+b` 后按 `d` 脱离会话并保持进程运行。

录制默认关闭。需要录制时增加 `--data-exporter`，可选 `--dataset-name my_session`，启动器会用 `.venv_data_collection` 在独立 `recording` 窗口启动 exporter，录制任务描述使用启动时的 prompt。用 `Ctrl+b` 后按 `n` 切换窗口；在键盘窗格输入 `c` 开始/停止录制，`x` 丢弃当前 episode。现有 exporter 不处理 `s`/`f` 或运行时 prompt 更新。

需要显示实时 RGB 时增加 `--camera-viewer`：

```bash
python3 gear_sonic/scripts/my_launch.py \
    --model-path /path/to/checkpoint \
    --camera-host 192.168.123.164 \
    --prompt "pour water into the cup" \
    --camera-viewer
```

查看器连接与推理客户端相同的相机地址，在桌面弹出 `SONIC Camera Viewer`，日志放在 tmux 的 `camera` 窗口。它使用 `.venv_data_collection` 的 GUI 版 OpenCV；Isaac 环境中的 headless OpenCV 无法显示窗口。需从有桌面显示的终端运行。在 RGB 窗口内按 `q` 关闭查看器；操作机器人仍在键盘发布器窗格内进行。

其他参数包括 `--device`、`--camera-port`、`--policy-port`、`--interface` 和 `--session`。`--deploy-checkpoint` 指定 SONIC 模型前缀，`--deploy-obs-config` 指定 observation 配置，两者的相对路径以 `gear_sonic_deploy/` 为基准。动作、状态、键盘通路固定使用本机 `5556/5557/5580`。`--dry-run` 只打印启动命令，不检查依赖或创建会话；`--help` 列出全部参数。

已有同名会话时，启动器直接附加，不重启推理或控制进程；增加 `--camera-viewer` 可以补开查看器，已关闭的查看器也会重新启动，正在运行的查看器不重复创建。其余启动参数不应用到已有进程。重连和关闭：

```bash
tmux attach -t '=my_g1_inference'
tmux kill-session -t '=my_g1_inference'
```

### 手动分终端启动

在 GPU 主机的 Isaac-GR00T 环境启动模型服务器：

```bash
cd /home/xhchen/code/Isaac-GR00T
.venv/bin/python -m gr00t.eval.run_gr00t_server \
    --model-path /path/to/checkpoint \
    --embodiment-tag MY_G1 \
    --device cuda:0 --host 0.0.0.0 --port 5550
```

在 SONIC 主机启动匹配训练数据的 C++ controller。下面使用 SONIC v1.1 的模型与 observation 配置；模型前缀应指向本机对应的部署文件：

```bash
cd /home/xhchen/code/GR00T-WholeBodyControl/gear_sonic_deploy
./deploy.sh \
    --cp policy/sonic_v1_1/model \
    --obs-config policy/sonic_v1_1/observation_config.yaml \
    --input-type zmq_manager --output-type zmq real
```

相机服务发布 `ego_view`，C++ 发布 `g1_debug` 状态。相机设置见 [Data collection](data_collection.md)。在单独的终端启动键盘发布器，输入键名后回车；输入 `t 新指令` 更新 prompt：

```bash
/home/xhchen/code/Isaac-GR00T/.venv/bin/python -c '
import time
import zmq

context = zmq.Context()
publisher = context.socket(zmq.PUB)
publisher.bind("tcp://127.0.0.1:5580")
time.sleep(0.5)
while True:
    key = input("k/i/p/[/]/c/s/f or t <prompt>: ")
    publisher.send_string("prompt:" + key[2:] if key.startswith("t ") else key)
'
```

在同时具备 Isaac-GR00T 与 SONIC 依赖的环境启动客户端。本机验证使用以下环境和命令：

```bash
cd /home/xhchen/code/GR00T-WholeBodyControl
PYTHONPATH=/home/xhchen/code/GR00T-WholeBodyControl \
    /home/xhchen/code/Isaac-GR00T/.venv/bin/python \
    -m gear_sonic.scripts.run_my_g1_vla_inference \
    --host 127.0.0.1 --port 5550 \
    --camera-host 192.168.123.164 \
    --state-zmq-host 127.0.0.1 \
    --prompt "pour water into the cup"
```

`--help` 列出网络与控制选项。客户端默认 `--action-horizon 0`，表示从服务器 `action.delta_indices` 自动取得 horizon。显式设置正数时，该数值必须与服务器一致。默认动作发布频率为 50 Hz，推理请求频率为 2.5 Hz。

| 通路 | 默认端口 | 客户端行为 |
| --- | --- | --- |
| PolicyServer | 5550 | REQ/REP：读取配置、请求动作 |
| 相机 | 5555 | SUB：读取当前 ego RGB 帧 |
| C++ 状态 | 5557 | SUB：读取 `g1_debug` |
| C++ 动作 | 5556 | PUB：绑定地址，发送 `pose` / `command` |
| 键盘发布器 | 5580 | SUB：接收控制键与 prompt |

跨主机部署时，C++ 动作订阅地址必须指向客户端 PUB 的地址。`--action-zmq-host` 是客户端绑定地址，其余 host 选项是客户端连接地址。

## 观测与动作约定

客户端调用 `PolicyClient.get_modality_config()`，选择服务器需要的 state 与 language 字段。MY_G1 tag 定义 embodiment；checkpoint 保存的 processor 配置定义该模型的输入字段、字段拼接顺序与 horizon。部署时由服务器加载 checkpoint 配置。

此入口提供当前帧 `video.ego_view`，形状为 `[1, 1, H, W, 3]`，使用 RGB；各 state 字段形状为 `[1, 1, D]`，language 字段为 `[[prompt]]`。支持的 state 字段为 `left_leg`、`right_leg`、`waist`、`left_arm`、`right_arm`、`left_hand`、`right_hand`、`projected_gravity`。服务器可选择其中的子集。入口使用 Locomanip 的单帧 ego 图像约定。

`body_q` 按 G1 的 29 个电机关节顺序进入 RobotModel；左右 `hand_q` 使用各自独立的 7 个 Dex3 实测电机值。RobotModel 按关节名字得到各 state 字段。与 Locomanip MY_G1 导出文件相对应的 43 维原始 `observation.state` 存储顺序为：

| 字段 | 原始存储切片 | 字段内顺序 |
| --- | --- | --- |
| left_leg | `[0:6]` | hip pitch, roll, yaw; knee; ankle pitch, roll |
| right_leg | `[6:12]` | hip pitch, roll, yaw; knee; ankle pitch, roll |
| waist | `[12:15]` | yaw, roll, pitch |
| left_arm | `[15:22]` | shoulder pitch, roll, yaw; elbow; wrist roll, pitch, yaw |
| left_hand | `[22:29]` | index 0, 1; middle 0, 1; thumb 0, 1, 2 |
| right_arm | `[29:36]` | shoulder pitch, roll, yaw; elbow; wrist roll, pitch, yaw |
| right_hand | `[36:43]` | index 0, 1; middle 0, 1; thumb 0, 1, 2 |

模型内部按 checkpoint 的 `state.modality_keys` 顺序拼接这些命名字段。上表描述的是数据文件的原始存储布局。

`base_quat` 为 `[w, x, y, z]`，客户端归一化后将世界重力 `[0, 0, -1]` 投影到机体坐标，得到 `projected_gravity`。

服务器输出每个动作块的 `motion_token [1, T, 64]`、`left_hand_joints [1, T, 7]` 和 `right_hand_joints [1, T, 7]`。它们分别来自训练数据的 `action.motion_token`、`action.left_hand` 和 `action.right_hand`。左右手动作顺序均为 **thumb 0, 1, 2; index 0, 1; middle 0, 1**，按此顺序直接发布。每个 protocol v4 pose 包包含一帧 64 维 token、左右 7 维手目标，以及递增的 frame index。token 的最大绝对值超过 1.25 时，当前推理结果被过滤。

初始 pose 使用 `gear_sonic/utils/inference/initial_poses.py` 的 `LATENT_INITIAL_MOTION_TOKEN`。该文件、SONIC 部署模型及初始手目标依赖由使用者维护，并与数据合成时的 SONIC 配置匹配。

## 操作流程

客户端启动后处于暂停发布状态，相机与状态可用时持续执行异步推理。C++ controller 的运行状态由控制命令管理。

1. 按 `k` 启动 C++，进入 PLANNER 模式。
2. 按 `i` 切换到 POSE 模式并发送初始 token。存在已发送 token 时，按 `--initial-pose-blend-duration` 平滑插值，默认 1 秒；首次发送初始 token 后等待 1 秒。此操作暂停策略发布并重置动作缓存与帧计数。
3. 确认机器人在初始姿态稳定后，按 `p` 恢复策略动作发布。

| 输入 | 行为 |
| --- | --- |
| `p` | 暂停 / 恢复策略动作发布；异步推理继续 |
| `k` | 启动 / 停止 C++ 控制循环 |
| `i` | 切换初始 pose，暂停策略发布 |
| `[` / `]` | 切换下一次初始 pose 的左 / 右手开合 |
| `prompt:<text>` | 更新后续推理的语言指令；键盘发布器通过 `t` 输入文本 |
| `c` / `s` / `f` | 显示录制提示，由已启动的数据 exporter 执行录制 |
| Ctrl+C | 关闭客户端线程与通信资源 |

`p` 与 Ctrl+C 管理客户端，C++ 的停止由 `k` 或部署端的停止操作控制。机器人站立由 SONIC C++ controller 执行，客户端提供初始 token 与后续策略目标。

## 本机验证

验证使用隔离端口与本机 ZMQ 发布 / 订阅进程。真实 PolicyServer 搭配确定性 MY_G1 policy 的 horizon 为 8；观测顺序、动作逐值传输、暂停推理、键盘控制、prompt、初始 pose 和 horizon 匹配检查均通过。2.2 秒收到 110 条策略 pose，实测发布频率约 49.7 Hz。

当前 Locomanip 合成 episode 的实际帧与导出 Parquet 对照：七组关节值完全一致，重力最大绝对误差约 `1.43e-8`；token 和手动作在 protocol v4 打包解包后完全一致。协议验证覆盖 Python 客户端与 ZMQ 通路。机器人稳定性与任务执行效果需要使用对应 SONIC 模型在部署环境验证。

本地 `pour_water_no_third_374ep_bs64_seed5218_0.3drop` 权重通过实际观测的模型推理，返回 token `[1, 40, 64]` 和双手 `[1, 40, 7]`，所有值有限且 token 最大绝对值约 0.539。该 checkpoint 保存了当前本地 processor 构造函数无法读取的 `state_noise_std=0.02`，原路径直接加载会报错；这次模型验证使用 `/tmp` 下的 checkpoint 副本，设置本地 backbone 路径并省略该训练参数。正式部署应使用与 checkpoint 对应的 processor 版本。checkpoint 源文件保持原样。

完整客户端进程连接上述服务器，使用实际合成帧作为相机和状态来源，自动读取 horizon 40。恢复发布后 6 秒收到 295 条动作消息，实测约 49.74 Hz，帧号连续 `0–294`，18 个推理块成功，worker 错误为 0，所有 protocol v4 字段为有限值。该验证使用本机动作订阅进程接收客户端输出。
