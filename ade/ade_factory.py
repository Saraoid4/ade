import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ade.arbitrated_ensemble_abstaining_threshold_diversity import ArbitratedEnsembleAbstainingThresholdDiversity
from ade.arbitrated_ensemble_abstaining_threshold_simple import ArbitratedEnsembleAbstainingThresholdSimple
from ade.arbitrated_ensemble_abstaining_prob import ArbitratedEnsembleAbstainingProb
from ade.arbitrated_ensemble_abstaining_prob_diversity import ArbitratedEnsembleAbstainingProbDiversity
from ade.arbitrated_ensemble_abstaining_twice_threshold_simple import ArbitratedEnsembleAbstainingTwiceThresholdSimple
from ade.arbitrated_ensemble_abstaining_twice_prob import ArbitratedEnsembleAbstainingTwiceProb
from ade.arbitrated_ensemble_abstaining_tradeoff_accu_div import ArbitratedEnsembleAbstainingTradeoff
from ade.arbitrated_ensemble_abstaining_relevance_rendundancy import ArbitratedEnsembleAbstainingRelevanceRedundancy
from ade.arbitrated_ensemble_abstaining_percentage import ArbitratedEnsembleAbstainingPercentage
from ade.arbitrated_ensemble_abstaining_best import ArbitratedEnsembleAbstainingBest
from ade.simple_ensemble import SimpleEnsemble
from ade.arbitrated_ensemble import ArbitratedEnsemble


class ArbitratedEnsembleFactory:

    def get_ensemble(self, abstain=None, abstain_twice=None, threshold_selection=None, probability_selection=None,
                     diversity_selection=None, naive=None, percentage_selection=False, n_best_selection=None,
                     mmr_criterion_selection=None, args_dict=None):
        # we get valid combinations only
        if abstain:
            if mmr_criterion_selection:
                if args_dict['trade_off'] == 'accu_div':
                    return ArbitratedEnsembleAbstainingTradeoff(**args_dict)
                elif args_dict['trade_off'] == 'relevance_redundancy':
                    return ArbitratedEnsembleAbstainingRelevanceRedundancy(**args_dict)
                else:
                    raise ValueError('Wrong trade off combination')

            if percentage_selection:
                return ArbitratedEnsembleAbstainingPercentage(**args_dict)

            if threshold_selection:
                if diversity_selection:
                    return ArbitratedEnsembleAbstainingThresholdDiversity(**args_dict)
                else:
                    return ArbitratedEnsembleAbstainingThresholdSimple(**args_dict)

            if probability_selection:
                if diversity_selection:
                    return ArbitratedEnsembleAbstainingProbDiversity(**args_dict)
                else:
                    return ArbitratedEnsembleAbstainingProb(**args_dict)
            if n_best_selection:
                return ArbitratedEnsembleAbstainingBest(**args_dict)

        elif abstain_twice:
            if threshold_selection:
                return ArbitratedEnsembleAbstainingTwiceThresholdSimple(**args_dict)
            if probability_selection:
                return ArbitratedEnsembleAbstainingTwiceProb(**args_dict)

        elif naive:
            return SimpleEnsemble(base_models=args_dict['base_models'])
        elif not naive:
            # del args_dict['meta_confidence_level']
            return ArbitratedEnsemble(**args_dict)
        else:
            raise ValueError('Invalid input combination')
