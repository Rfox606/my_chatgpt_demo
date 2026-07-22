import numpy as np
import pandas as pd

from continuous_state_v2.config import ContinuousStateV2Config
from continuous_state_v2.online_forecast import run_online_forecasts, train_frozen_predictors


def test_forecast_updates_only_when_observation_arrives():
    c=ContinuousStateV2Config(forecast_history_windows=5); n=260; cycle=np.arange(n)*5.
    frame=pd.DataFrame({"dataset":"t","window_index":range(n),"center_cycle":cycle,"P_common":cycle/1000,"BD":cycle/2000,"B_terminal":cycle/3000,"P_RS20":0.,"P_RS50":0.,"BD_RS20":0.,"BD_RS50":0.,"B_RS20":0.,"TES":0.,"weighted_oos_common":0.})
    frozen=train_frozen_predictors(frame,c); pred,_=run_online_forecasts(frame,frozen,"d",c)
    seen=pred[pred.observation_available.eq(1)]
    assert not seen.empty and seen.online_model_updated_after_observation.eq(1).all()
