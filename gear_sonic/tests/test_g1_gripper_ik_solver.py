import numpy as np
import pytest

from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
    G1GripperInverseKinematicsSolver,
)

THUMB_0_TARGET = 0.481710873


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("trigger", [0.0, 0.1, 0.199, 0.2])
def test_retarget_from_trigger_keeps_fingers_open_below_deadzone(side, trigger):
    solver = G1GripperInverseKinematicsSolver(side)

    q_desired = solver.retarget_from_trigger(trigger)

    assert q_desired[0] == pytest.approx(THUMB_0_TARGET)
    np.testing.assert_allclose(q_desired[1:], np.zeros(6))


@pytest.mark.parametrize("side", ["left", "right"])
def test_retarget_from_trigger_half_closes_at_point_six(side):
    solver = G1GripperInverseKinematicsSolver(side)
    q_closed = solver._get_middle_close_q_desired()

    q_desired = solver.retarget_from_trigger(0.6)

    assert q_desired[0] == pytest.approx(THUMB_0_TARGET)
    np.testing.assert_allclose(q_desired[1:], 0.5 * q_closed[1:])


@pytest.mark.parametrize("side", ["left", "right"])
def test_retarget_from_trigger_matches_closed_pose_at_one(side):
    solver = G1GripperInverseKinematicsSolver(side)
    q_closed = solver._get_middle_close_q_desired()

    q_desired = solver.retarget_from_trigger(1.0)

    assert q_desired[0] == pytest.approx(THUMB_0_TARGET)
    np.testing.assert_allclose(q_desired[1:], q_closed[1:])
