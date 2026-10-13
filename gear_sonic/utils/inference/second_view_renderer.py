"""Robot-only Warp rendering, copied from Locomanip's second-person pipeline.

Camera placement, appearance and render settings follow rendering/camera.py,
rendering/mujoco.py and rendering/warp.py in humanoid-locomanip. This standalone
snapshot uses the SONIC model assets and measured motor ordering.
"""

import json
import math
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
HAND_JOINT_SUFFIXES = (
    "thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1",
)


class SecondViewRenderer:
    """Reconstruct measured G1 poses and render RGB on white with MuJoCo Warp."""

    def __init__(self, model_path: str | Path, camera_config_path: str | Path, device="cuda:0"):
        self.camera_config = json.loads(Path(camera_config_path).read_text())
        spec = mujoco.MjSpec.from_file(str(model_path))
        spec.worldbody.add_camera(name="second_view")
        self.model = spec.compile()
        self.model.opt.timestep = 0.005

        # Locomanip configure_render_appearance snapshot.
        self.model.vis.headlight.diffuse[:] = 0.40
        self.model.vis.headlight.ambient[:] = 0.35
        self.model.vis.headlight.specular[:] = 0.0
        self.model.light_diffuse[:] = 0.48
        self.model.light_ambient[:] = 0.0
        self.model.light_specular[:] = 0.10
        for geom_id, body_id in enumerate(self.model.geom_bodyid):
            body_name = self.model.body(int(body_id)).name
            is_hand = body_name.startswith(("left_hand_", "right_hand_"))
            if not is_hand and self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_name = self.model.mesh(int(self.model.geom_dataid[geom_id])).name
                is_hand = mesh_name.startswith(("left_hand_", "right_hand_"))
            if is_hand:
                self.model.geom_rgba[geom_id] = (0.12, 0.12, 0.12, 1.0)

        self.data = mujoco.MjData(self.model)
        self.root_address = int(self.model.joint("floating_base_joint").qposadr[0])
        self.joint_addresses = {
            key: np.array([int(self.model.joint(name).qposadr[0]) for name in names])
            for key, names in (
                ("body_q", BODY_JOINT_NAMES),
                ("left_hand_q", [f"left_hand_{suffix}_joint" for suffix in HAND_JOINT_SUFFIXES]),
                ("right_hand_q", [f"right_hand_{suffix}_joint" for suffix in HAND_JOINT_SUFFIXES]),
            )
        }
        self.camera_id = self.model.camera("second_view").id
        config = self.camera_config
        self.width, self.height = config["width"], config["height"]
        self.model.cam_sensorsize[self.camera_id] = (self.width, self.height)
        self.model.cam_intrinsic[self.camera_id] = (
            config["fx"], config["fy"], self.width / 2 - config["cx"],
            self.height / 2 - config["cy"],
        )
        self.model.cam_resolution[self.camera_id] = (self.width, self.height)
        mujoco.mj_forward(self.model, self.data)

        subtree = {self.model.body("pelvis").id}
        while True:
            descendants = {
                body_id for body_id, parent_id in enumerate(self.model.body_parentid)
                if int(parent_id) in subtree
            }
            if descendants <= subtree:
                break
            subtree.update(descendants)
        robot_ids = [
            geom_id for geom_id, body_id in enumerate(self.model.geom_bodyid)
            if int(body_id) in subtree and self.model.geom(geom_id).name != "floor"
        ]
        foreground = np.zeros(self.model.ngeom, dtype=bool)
        foreground[robot_ids] = True
        self.foreground = torch.as_tensor(foreground, device=device)

        wp.init()
        index = torch.device(device).index
        self.warp_device = wp.get_device(f"cuda:{index if index is not None else torch.cuda.current_device()}")
        with wp.ScopedDevice(self.warp_device):
            self.gpu_model = mjw.put_model(self.model)
            self.gpu_data = mjw.make_data(self.model, nworld=2, nconmax=0, njmax=0)
            geom_groups = self.model.geom_group.copy()
            self.model.geom_group[:] = 3
            self.model.geom_group[robot_ids] = geom_groups[robot_ids]
            try:
                self.context = mjw.create_render_context(
                    self.model, nworld=2, cam_active=[self.camera_id],
                    cam_res=[(self.width, self.height)],
                    render_rgb=True, render_depth=True, render_seg=True,
                    use_textures=True, use_shadows=True,
                    background_color=(1.0, 1.0, 1.0, 1.0), samples_per_pixel=1,
                )
            finally:
                self.model.geom_group[:] = geom_groups
        self.host = {
            name: np.empty((2,) + getattr(self.data, name).shape, np.float32)
            for name in ("geom_xpos", "geom_xmat", "cam_xpos", "cam_xmat", "light_xpos", "light_xdir")
        }
        self.rgb_start = int(self.context.rgb_adr.numpy()[0])
        self.seg_start = int(self.context.seg_adr.numpy()[0])

    def render(self, state: dict) -> np.ndarray:
        """Return uint8 RGB for the measured motor arrays and wxyz base quaternion."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.root_address + 3:self.root_address + 7] = state["base_quat"]
        for key, addresses in self.joint_addresses.items():
            self.data.qpos[addresses] = state[key]
        mujoco.mj_forward(self.model, self.data)

        # Locomanip _BoundThirdPersonCamera.pose snapshot; translation follows root.
        root = self.data.qpos[self.root_address:self.root_address + 7]
        w, x, y, z = root[3:7] / np.linalg.norm(root[3:7])
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        rotation = np.array(((math.cos(yaw), -math.sin(yaw)), (math.sin(yaw), math.cos(yaw))))
        config = self.camera_config
        position = root[:3].copy()
        position[:2] += rotation @ np.asarray(config["camera_offset_m"][:2])
        position[2] += config["camera_offset_m"][2]
        look_at = root[:3].copy()
        look_at[:2] += rotation @ np.asarray(config["look_at_offset_m"][:2])
        look_at[2] += config["look_at_offset_m"][2]
        forward = look_at - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, (0.0, 0.0, 1.0))
        right /= np.linalg.norm(right)
        back = -forward
        camera_rotation = np.column_stack((right, np.cross(back, right), back))

        for name, host in self.host.items():
            host[:] = getattr(self.data, name)
        self.host["cam_xpos"][:, self.camera_id] = position
        self.host["cam_xmat"][:, self.camera_id] = camera_rotation.reshape(9)
        with wp.ScopedDevice(self.warp_device):
            for name, host in self.host.items():
                getattr(self.gpu_data, name).assign(host)
            wp.synchronize_device(self.warp_device)
            mjw.refit_bvh(self.gpu_model, self.gpu_data, self.context)
            mjw.render(self.gpu_model, self.gpu_data, self.context)
            wp.synchronize_device(self.warp_device)

        n = self.width * self.height
        packed = wp.to_torch(self.context.rgb_data)[0, self.rgb_start:self.rgb_start + n]
        rgb = packed.view(torch.uint8).reshape(self.height, self.width, 4)[..., :3].flip(-1)
        ids = wp.to_torch(self.context.seg_data)[0, self.seg_start:self.seg_start + n]
        ids = ids.reshape(self.height, self.width, 2)
        foreground = (
            (ids[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
            & self.foreground[ids[..., 0].clamp_min(0).long()]
        )
        return rgb.masked_fill(~foreground[..., None], 255).cpu().numpy()

    def close(self):
        """Release the CUDA render context and model buffers."""
        wp.synchronize_device(self.warp_device)
        self.context = None
        self.gpu_data = None
        self.gpu_model = None
