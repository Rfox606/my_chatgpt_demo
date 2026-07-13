import numpy as np
from ordered_regime_v21.regime_memory import StateMemory

def test_trimmed_memory_resists_single_outlier():
    state=StateMemory(1,np.zeros(2),np.zeros(2),1.,1.)
    for index in range(5): state.add({"level":np.array([.1,0.]),"trajectory":np.array([.1,0.]),"confidence":1.},20,1,True)
    state.add({"level":np.array([100.,0.]),"trajectory":np.array([100.,0.]),"confidence":1.},20,1,True)
    assert state.level_proto[0] < 1
