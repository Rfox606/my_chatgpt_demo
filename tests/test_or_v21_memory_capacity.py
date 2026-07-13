import numpy as np
from ordered_regime_v21.regime_memory import StateMemory

def test_memory_has_fixed_capacity():
    state=StateMemory(1,np.zeros(2),np.zeros(2),1.,1.)
    for index in range(10): state.add({"level":np.zeros(2),"trajectory":np.zeros(2),"confidence":float(index)},3,99,True)
    assert state.support==3
