import numpy as np
from tests.or_v21_test_utils import engine,stream
def test_prefix_outputs_are_exactly_causal():
    frame,encoded=stream([0]*8+[4]*8);full,_,_,_=engine().run(frame,encoded,np.zeros(len(frame)));short,_,_,_=engine().run(frame.iloc[:8].copy(),{k:v[:8] for k,v in encoded.items()},np.zeros(8))
    assert np.array_equal(full.loc[:7,"predicted_stage"].to_numpy(),short.predicted_stage.to_numpy())
