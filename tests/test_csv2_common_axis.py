import pandas as pd

from continuous_state_v2.common_axis import build_common_axis
from continuous_state_v2.config import ContinuousStateV2Config


def test_common_axis_uses_only_stable_same_sign_features():
    s=pd.DataFrame([
        {"direction_id":"Exp1_source","feature_name":"a","median_weight":.2,"sign_stability":.9},
        {"direction_id":"Exp2_source","feature_name":"a","median_weight":.1,"sign_stability":.9},
        {"direction_id":"Exp1_source","feature_name":"b","median_weight":.2,"sign_stability":.9},
        {"direction_id":"Exp2_source","feature_name":"b","median_weight":-.1,"sign_stability":.9},
    ])
    w, features, table, status=build_common_axis(s,{"a","b"},ContinuousStateV2Config())
    assert features == ("a",) and status == "LOW_DIMENSION_SUPPORT" and w[0] > 0
    assert table.loc[table.feature_name.eq("b"),"kept_common"].iloc[0] == 0
