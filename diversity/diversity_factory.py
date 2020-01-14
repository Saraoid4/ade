from .diversity_measures import *
from .dissimilarity_matrix import DissimilarityMatrix
from .symmetric_uncertainty import SymmetricUncertainty

class DiversityMeasuresFactory:

    def get_diversity_evaluator(self, diversity_method=None, diversity_measure=None, args_dict=None):
        if diversity_method == 'sliding_window':
            if diversity_measure == 'correlation':
                return WindowCorrelationMeasure(**args_dict)
            elif diversity_measure == 'entropy':
                return WindowEntropyMeasure(**args_dict)
            elif diversity_measure == 'disagree':
                return WindowDisagreementEvaluator(**args_dict)
            elif diversity_measure == 'double_fault':
                return WindowDoubleFaultEvaluator(**args_dict)
            elif diversity_measure == 'dissimilarity':
                return DissimilarityMatrix(**args_dict)
            elif diversity_measure == 'redundancy':
                return SymmetricUncertainty(**args_dict)
            else:
                raise ValueError('Not Handleled combination:', diversity_method, diversity_measure)

        elif diversity_method == 'fading factor':
            if diversity_measure == 'disagree':
                return DisagreementEvaluator(**args_dict)
            elif diversity_measure == 'double_fault':
                return DoubleFaultEvaluator(**args_dict)
            else:
                raise ValueError('Not Handleled combination:', diversity_method, diversity_measure)

        elif diversity_method == 'incremental':
            if diversity_measure == 'disagree':
                return IncrementalDisagree(**args_dict)
            elif diversity_measure == 'double_fault':
                return IncrementalDoubleFault(**args_dict)
            else:
                raise ValueError('Not Handleled combination:', diversity_method, diversity_measure)
        else:
            raise ValueError('diversity methods must be ')









