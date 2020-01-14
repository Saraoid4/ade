import numpy as np
from skmultiflow.core.base import StreamModel
from skmultiflow.utils import get_dimensions
from skmultiflow.utils import FastBuffer, FastComplexBuffer
from random import randint
from ade.file_utils import init_file, update_file
from os.path import join



from utils.metrics import get_predictions_errors

class BLAST:

    def __init__(self, base_models, window_size, alpha, loss_function="MSE", output_file=None):
        
        super().__init__()

        self.base_models = base_models
        self.window_size = window_size
        self.alpha = alpha
        self.loss_function = loss_function

        # Keep in memory current active model for predictions
        self.active_models_idx = randint(0, len(self.base_models)-1)

        # keep in memory previous base-predictions and true values
        self.last_window_predictions = FastComplexBuffer(max_size=self.window_size, width=len(base_models))
        self.last_window_true_value = FastBuffer(max_size=self.window_size) 

        # Index of the current example index 
        self.current_example_idx = 0

        self.first_run = True 
        # output_file = join(output_file, self.get_model_name() + '.csv')
        if output_file is not None: 
            self.output_file = join(output_file, self.get_model_name() + '.csv')
            base_models_info = [b_model.get_info() for b_model in self.base_models]
            init_file(output_file=self.output_file, ensemble_size=len(self.base_models),base_models_info=base_models_info, global_info=self.get_info())
        else: 
            self.output_file= None


    def partial_fit(self, X, y):
        
        for b_model in self.base_models:
            b_model.partial_fit(X, y)   
        # TODO : check best place to update
        self.current_example_idx += len(y)
        # Storing last true values
        self.last_window_true_value.add_element(y)
        
        if self.first_run:
            self.first_run = False
            r, _ = get_dimensions(X)
            for i in range(r):
                predictions = [b_model.predict([X[i]])[0] for b_model in self.base_models]
                self.last_window_predictions.add_element(predictions)
        
    def fit(self, X, y, classes=None, weight=None):
        raise NotImplementedError

    def predict(self, X):
        # Invoke active base_model
        final_prediction = self.base_models[self.active_models_idx].predict(X)
        
        # Get all base predcitions to upate file
        all_base_predictions = [b_model.predict(X)[0] for b_model in self.base_models]

       

        # Check if reached alpha processed samples and select base model to be used for the nex alpha instances 
        if self.current_example_idx% self.alpha == 0:
            # We have reached alpha processed samples
            new_active_idx = self._select_experts()
            self.active_models_idx = new_active_idx
        
        # Store all predictions to be used for evaluation        
        self.last_window_predictions.add_element(all_base_predictions)
        
        update_file(output_file=self.output_file, global_prediction=final_prediction[0], base_predictions=all_base_predictions, 
                meta_predictions=[0]* len(self.base_models), base_selected_idx=[self.active_models_idx])
              
        return final_prediction

    def _select_experts(self):
        
        # Compute performance on last w instances
        loss_vector = [0]* len(self.base_models)

        # Compute loss over last instances
        
        for i in range(self.last_window_predictions.current_size):
            predictions = self.last_window_predictions.get_queue()[i]
            true_value = self.last_window_true_value.get_queue()[i]
            
            step_loss_values = [get_predictions_errors(true_value, p, self.loss_function) for p in predictions]
            loss_vector = [a+b for a, b in zip(loss_vector, step_loss_values)]
        
        # Compute average loss on last instances
        loss_vector = [ x/self.window_size for x in loss_vector]
        
        # Retun model with minimum loss, if tie, random choose
        min_loss = min(loss_vector)
        minimum_loss_models_idx = [i for i in range(len(loss_vector)) if loss_vector[i] == min_loss]
        # Random choose between selected model
        selected_active_model_idx = minimum_loss_models_idx[randint(0, len(minimum_loss_models_idx)-1)]
        
        return selected_active_model_idx

    def _estimate_models_performance(self):
        # TODO: Check defintion of current example index 
        #for c in range(min(1, self.current_example_idx - self.window_size), self.current_example_idx):
        
        # TODO: if not to save ensemble size
        # This will store Loss induced by each base-model
        loss_vector = [0]* len(self.base_models)

        for i in range(1, self.window_size):
            # Get all base-models predictions at step i
            step_results = self.last_window_predictions[i]
            # Compute loss of each base model
            step_models_loss = [get_predictions_errors(self.last_window_true_value[i], step_results[j], metric=self.loss_function) for j in len(self.base_models)]
            # Update global loss vector 
            loss_vector = loss_vector + step_models_loss
    
    def get_info(self):
        info = '{}:'.format(type(self).__name__)
        info += ' - ensemble_size: {}'.format(len(self.base_models))
        info += ' - window_size:{}'.format(self.window_size)
        info += ' - alpha:{}'.format(self.alpha)
        info += ' - loss_function:{}'.format(self.loss_function)
        return info

    def predict_proba(self, X):
        raise NotImplementedError
    
    def reset(self):
        return NotImplementedError
    
    def score(self, X, y):
        raise NotImplementedError

    def get_class_type(self):
        return 'BLAST'

    def get_model_name(self):
        return 'BLAST_SW_{}_SIZE_{}'.format(self.window_size, str(len(self.base_models)))

    