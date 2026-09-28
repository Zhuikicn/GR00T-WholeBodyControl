import pytest

from decoupled_wbc.control.robot_model.supplemental_info.g1.g1_supplemental_info import (
    G1SupplementalInfo as DecoupledWbcG1SupplementalInfo,
)
from gear_sonic.data.robot_model.supplemental_info.g1.g1_supplemental_info import (
    G1SupplementalInfo as GearSonicG1SupplementalInfo,
)


@pytest.mark.parametrize(
    "info_type",
    (GearSonicG1SupplementalInfo, DecoupledWbcG1SupplementalInfo),
)
def test_right_hand_joint_limits_match_the_g1_urdf(info_type) -> None:
    limits = info_type().joint_limits

    assert limits["right_hand_thumb_0_joint"] == pytest.approx((-1.04719755, 1.04719755))
    assert limits["right_hand_thumb_1_joint"] == pytest.approx((-1.04719755, 0.72431163))
    assert limits["right_hand_thumb_2_joint"] == pytest.approx((-1.74532925, 0.0))
    assert limits["right_hand_index_0_joint"] == pytest.approx((0.0, 1.57079632))
    assert limits["right_hand_index_1_joint"] == pytest.approx((0.0, 1.74532925))
    assert limits["right_hand_middle_0_joint"] == pytest.approx((0.0, 1.57079632))
    assert limits["right_hand_middle_1_joint"] == pytest.approx((0.0, 1.74532925))
