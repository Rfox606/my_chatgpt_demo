import numpy as np
from adaptive_awr_v12.target_offset import make
def test_offsets_are_constants_and_bounded():
 s=np.arange(30.);t=s+3
 for mode in ("T0","T1","T2"):
  o=make(mode,s,t);z=np.array([1.,2.,3.]);assert np.allclose(np.diff(o.apply(z)),np.diff(z));assert abs(o.applied_shift)<=1
