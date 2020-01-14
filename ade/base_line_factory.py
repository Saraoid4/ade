from skmultiflow.meta import LeverageBagging, OzaBagging, OzaBaggingAdwin, AdaptiveRandomForest, RegressorChain
from skmultiflow.trees import RegressionHAT
from skmultiflow.lazy import KNN

LEVERAGE_BAG = 'leverage_bag'
OZA_BAG = 'oza_bag'
REGRESSOR_CHAIN = 'regressor_chains'
ARF = 'adaptive_random_forest'


class BASEFACTORY:
    
    @staticmethod
    def get_base_enemble(meta_ensemble, ensemble_size, base_estimator=RegressionHAT(), args=None):
        
        if meta_ensemble == LEVERAGE_BAG:
            ensemble = LeverageBagging(base_estimator=base_estimator, n_estimators=ensemble_size)
        elif meta_ensemble == OZA_BAG:
            ensemble = OzaBagging(base_estimator=base_estimator, n_estimators=ensemble_size)
        elif meta_ensemble == ARF: 
            ensemble = AdaptiveRandomForest(n_estimators=ensemble_size)
        elif meta_ensemble == REGRESSOR_CHAIN:
            ensemble = RegressorChain()
        else:
            raise ValueError('Invalid Meta-Ensemble Type {}'.format(meta_ensemble))
        return ensemble

