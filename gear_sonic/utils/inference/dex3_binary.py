"""Dex3 signed commands and teleoperation endpoints in thumb/index/middle order."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

DEX3_JOINT_SUFFIXES = (
    "thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1",
)
DEX3_CLOSURE_THRESHOLD = 0.30
DEX3_OPEN_QPOS = np.zeros(7)
DEX3_CLOSED_QPOS = {
    "left": np.array([0., .7, .7, -1., -1.5, -1., -1.5]),
    "right": np.array([0., -.7, -.7, 1., 1.5, 1., 1.5]),
}
for _qpos in (DEX3_OPEN_QPOS, *DEX3_CLOSED_QPOS.values()):
    _qpos.setflags(write=False)


def hand_command_from_qpos(qpos: ArrayLike, side: str) -> NDArray[np.int8]:
    """Classify measured thumb-first angles as open (-1) or closed (+1)."""
    delta = DEX3_CLOSED_QPOS[side] - DEX3_OPEN_QPOS
    threshold_pose = DEX3_OPEN_QPOS + DEX3_CLOSURE_THRESHOLD * delta
    projection = (np.asarray(qpos, dtype=np.float64) - threshold_pose) @ delta
    return np.where(projection >= 0., 1, -1).astype(np.int8)


def hand_qpos_from_command(command: ArrayLike, side: str) -> NDArray[np.float64]:
    """Expand signed commands to seven joint targets; the C++ driver limits them."""
    return np.where(np.asarray(command)[..., None] == 1, DEX3_CLOSED_QPOS[side], DEX3_OPEN_QPOS)
