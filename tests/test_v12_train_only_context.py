import numpy as np,pandas as pd
from adaptive_awr_v12.config import AdaptiveAWRV12Config
from adaptive_awr_v12.feature_support import support,candidate_sets
def test_validation_mutation_does_not_change_train_context():
 c=AdaptiveAWRV12Config();x=np.arange(100,dtype=float);train=pd.DataFrame({"AWR_adaptive":x,"BDall_xy_v2":x,"RS50_positive":x,"TES":x,"high_AWR_high_BD_occupancy":np.sin(x)+1});before=(support(train,c),candidate_sets(support(train,c)));validation=train.copy();validation.iloc[:]=999;after=(support(train,c),candidate_sets(support(train,c)));assert before[0].equals(after[0]) and before[1].equals(after[1])
