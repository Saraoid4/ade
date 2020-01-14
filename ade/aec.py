import numpy as np
from skmultiflow.core.base import StreamModel
from skmultiflow.utils import get_dimensions
from ade.file_utils import init_file, update_file
from os.path import join

class AEC(StreamModel):
    
    def __init__(self, base_models, fading_factor=0.950, output_file=None):
        super().__init__()
        self.base_models = base_models
        self.ensemble_size = len(self.base_models)
        
        # Forgetting factor
        self.fading_factor = fading_factor

        # Asymptotic memory N(t) = 1 + lambda* N(t-1)
        self.asynmptotic_memory = 0

        self.W_matrix = np.zeros([self.ensemble_size, self.ensemble_size], dtype=float)

        self.I_vector = [1] * self.ensemble_size
        self.D_vector = [1] * self.ensemble_size

        self.first_run = True
        self.output_file = join(output_file, self.get_model_name() + '.csv')
        base_models_info  = [b_model.get_info() for b_model in self.base_models]
        init_file(output_file=self.output_file, ensemble_size=len(self.base_models),base_models_info=base_models_info, global_info=self.get_info())
        


    def fit(self, X, y):
        raise NotImplementedError

    def partial_fit(self, X, y):
        if self.first_run:
            self.first_run = False
            for b_model in self.base_models:
                b_model.partial_fit(X, y)
            
        
        else:
            r, _ = get_dimensions(X)
            for i in range(r):
                # Update Vectors and matrices
                predictions = [b_model.predict([X[i]])[0] for b_model in self.base_models]
                # Compute error vector as a column
                true_value = y[i]
                errors_vector = np.array([true_value - pred for pred in predictions]).reshape((self.ensemble_size, 1))

                # Update W matrix 
                self.W_matrix = (errors_vector).dot(errors_vector.T) + self.fading_factor * self.W_matrix
                # Get current faded count
                N_t = self._set_asymptotoc_memory()
                omega_matrix = 1/N_t * self.W_matrix

                # Get estimator of v diagonal of omega
                v_estimator = np.diagonal(omega_matrix)
                
                
                # Update  new information Vector I
                self.I_vector = [(v_k_t_1**(-.5)) * np.exp(-.5 * (true_value - pred)**2  / v_k_t_1) for v_k_t_1, pred in zip(v_estimator, predictions)]

                # Update D vector 
                self.D_vector = [I_k_t_1 * D_k_t_1**self.fading_factor for I_k_t_1, D_k_t_1 in zip(self.I_vector, self.D_vector)]
                
                for b_model in self.base_models:
                    b_model.partial_fit([X[i]], [y[i]])

    def predict(self, X):
        # Compute base_learners weights 
        weights = [D_k_t / sum(self.D_vector) for D_k_t in self.D_vector]
        # print('weights{}\n'.format(weights))
        predictions = [b_model.predict(X)[0] for b_model in self.base_models]
        # Return a weighted average of all predictions 
        final_prediction = [np.average(predictions, weights=weights)]

       

        update_file(output_file=self.output_file, global_prediction=final_prediction[0], base_predictions=predictions, 
                    meta_predictions=[0]* len(self.base_models), base_selected_idx=list(range(len(self.base_models))))
        return final_prediction

    def predict_proba(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_info(self):
        info = '{}:'.format(type(self).__name__)
        info += ' - ensemble_size: {}'.format(self.ensemble_size)
        info += ' - fading_factor:{}'.format(self.fading_factor)
        return info

    def get_model_name(self):
        return 'AEC_FF_{}_SIZE_{}'.format(self.fading_factor, self.ensemble_size)
    
    def _set_asymptotoc_memory(self):
        self.asynmptotic_memory = 1 + self.fading_factor * self.asynmptotic_memory
        return self.asynmptotic_memory

    
    
    
