# MY_G1 VLA inference with SONIC

`gear_sonic/scripts/run_my_g1_vla_inference.py` 是 Locomanip MY_G1 模型的独立推理入口。它从 Isaac-GR00T PolicyServer 读取 checkpoint 的有效 modality 配置，订阅 ego 相机，并按模型要求订阅实时渲染的 `second_view`，向 SONIC C++ 发布 latent protocol v4 动作。

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

相机服务需在机器人上运行并提供 `ego_view`，设置见 [Data collection](data_collection.md)。需要 `second_view` 的模型还需单独运行下文的渲染程序。启动器负责服务器、C++ 和 client；相机与 second-view 渲染服务独立启动。`--model-path` 必须指向可由当前 Isaac-GR00T 环境加载的本地 checkpoint 目录。

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
| second_view 渲染 | 5560 | 模型需要时 SUB：读取最新白底机器人 RGB 帧 |
| C++ 状态 | 5557 | SUB：读取 `g1_debug` |
| C++ 动作 | 5556 | PUB：绑定地址，发送 `pose` / `command` |
| 键盘发布器 | 5580 | SUB：接收控制键与 prompt |

跨主机部署时，C++ 动作订阅地址必须指向客户端 PUB 的地址。`--action-zmq-host` 是客户端绑定地址，其余 host 选项是客户端连接地址。

### 独立 second_view 渲染服务

`gear_sonic/scripts/run_second_view_renderer.py` 订阅 C++ 的 `g1_debug`，使用身体、双手实测电机关节和 `[w,x,y,z]` 的 `base_quat` 重建机器人姿态，经 MuJoCo Warp 渲染后发布图像。进程执行姿态前向计算，独立于机器人的物理仿真和控制循环。

渲染实现位于 `gear_sonic/utils/inference/second_view_renderer.py`。这是当前 Locomanip robot-only 渲染逻辑的独立副本，运行时依赖 SONIC 机器人资产。默认模型为 `gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml`，相机参数独立保存在 `gear_sonic/config/second_view_camera.json`，供使用者维护。

渲染环境需要 CUDA、PyTorch、MuJoCo 3.14.0、mujoco-warp 3.14.0、warp-lang 1.15.0，以及 NumPy、OpenCV、tyro、pyzmq、msgpack、msgpack-numpy。本机验证使用已有的 Locomanip `.venv`，该解释器提供渲染依赖；代码通过 `PYTHONPATH` 从 WBC 加载：

```bash
cd /home/xhchen/code/GR00T-WholeBodyControl
PYTHONPATH=/home/xhchen/code/GR00T-WholeBodyControl \
    /home/xhchen/code/humanoid-locomanip/.venv/bin/python \
    -m gear_sonic.scripts.run_second_view_renderer \
    --state-host 127.0.0.1 --state-port 5557 \
    --publish-host 0.0.0.0 --publish-port 5560 \
    --fps 50 --device cuda:0
```

可在具备上述依赖的独立环境运行同一命令。使用 `--model-path`、`--camera-config-path` 指定模型和相机参数。进程等待首个 state 后开始渲染，始终处理最新消息，目标发布频率默认 50 Hz，每 5 秒报告实际 FPS、渲染和 PNG 发布耗时。

画面为 1280×720 白底机器人，使用当前合成数据的材质、灯光、纹理、阴影及单样本采样设置。相机内参为 `fx=fy=429.03129333391564`、主点 `(640,360)`；相机偏移 `(0.6,0,0.5)`、观察点偏移 `(0.1,0,0)`，水平偏移随机器人 yaw 旋转。根位置采用模型默认值，相机随根位置跟随。身体和双手按关节名字写入 MJCF，Dex3 电机顺序为 thumb、index、middle。

发布消息沿用 `ImageMessageSchema`，图像键为 `second_view`，图像内容为 PNG bytes，通过单条 msgpack/ZMQ 消息传输。PNG 编码输入为 BGR，订阅器解码后恢复 RGB。时间戳 `timestamps.second_view` 表示渲染器收到源 state 的本机时间；`metadata.source_state_index` 标记源 state，`metadata.rendered_at` 表示渲染编码完成时间。

client 的 `--second-view-host`、`--second-view-port` 默认 `localhost:5560`；跨主机部署时将 host 设为渲染主机地址。client 仅在服务器需要 `second_view` 时建立订阅。渲染服务可以与模型服务器使用不同主机或 GPU。

## 观测与动作约定

客户端调用 `PolicyClient.get_modality_config()`，选择服务器需要的 state 与 language 字段。MY_G1 tag 定义 embodiment；checkpoint 保存的 processor 配置定义该模型的输入字段、字段拼接顺序与 horizon。部署时由服务器加载 checkpoint 配置。

此入口提供当前帧 `video.ego_view`，并按服务器配置提供 `video.second_view`。每路视频形状为 `[1, 1, H, W, 3]`，使用 RGB；各 state 字段形状为 `[1, 1, D]`，language 字段为 `[[prompt]]`。支持的 state 字段为 `left_leg`、`right_leg`、`waist`、`left_arm`、`right_arm`、`left_hand`、`right_hand`、`projected_gravity`。服务器可选择其中的子集。

`second_view` 对应合成数据的白底机器人视频 `camera/second_person/rgb.mp4`，数据集存储键为 `observation.images.second_person`。带场景背景的 `third_view` 用于另一种视觉约定。

双视角推理在所需图像和 state 首次到达后开始。ego、second_view 和关节 state 各自缓存最新样本，独立更新；client 采用当前最新关节 state，并与最近的渲染图像组合。图像订阅器沿用最新帧缓存行为，服务中断后可能重复使用缓存图像。此入口提供单帧视觉观测。

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

初始 pose 使用 `gear_sonic/utils/inference/initial_poses_v1_1.py` 的 `LATENT_INITIAL_MOTION_TOKEN`。该文件、SONIC 部署模型及初始手目标依赖由使用者维护，并与数据合成时的 SONIC 配置匹配。

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

本机验证使用隔离端口、真实合成的 walking / approach / manipulation 姿态，以及本地 ZMQ 发布订阅进程。

| 验证 | 实测结果 |
| --- | --- |
| 9 个姿态，与原 robot-only Warp 渲染使用相同固定根位置比较 | qpos 与 RGB 逐值完全一致 |
| 50 Hz state 回放，398 帧 PNG | 全帧解码像素相同，稳定 49.52 FPS |
| 源 state 发布至图像接收 | 中位 31.72 ms，P95 38.02 ms |
| 渲染器收到 state 至图像接收 | 中位 16.78 ms，P95 19.11 ms |
| 预热后的直接渲染 | 中位 2.93 ms，P95 3.26 ms |
| 渲染进程 GPU 显存 | 790 MiB |
| 单 ego、双视角等待、最新 state 与图像、horizon、键盘与动作播放 | 实际 PolicyServer / ZMQ 验证通过 |

固定根位置与原 episode 根位置的参考图像比较，两个姿态分别有 14 和 11 个边缘像素出现浮点光线追踪差异；构图与相机跟随保持一致。

实际双视角 checkpoint `pour_water_third_joint_bs64_seed5218_0.5drop` 使用新 renderer 和完整 client，服务器返回 `ego_view + second_view`、horizon 40。token `[1,40,64]` 和双手 `[1,40,7]` 全部为有限值；直接推理 token 最大绝对值约 0.496。恢复播放后 6 秒收到 286 条连续策略动作，约 48.98 Hz，12 个推理块，token 过滤跳过次数为 0。服务器验证使用临时本地 backbone 路径，checkpoint 原文件保持原样。进程正常退出。

这些结果验证了渲染、通信和权重推理。机器人站立与任务效果需在对应 SONIC 部署环境验证。
