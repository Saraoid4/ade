import numpy as np
from os.path import join


from skmultiflow.core.base import StreamModel
from skmultiflow.utils import FastComplexBuffer
from ade.file_utils import init_file, update_file

class MAB(StreamModel):

    def __init__(self, base_models, epsilon, fading_factor=0.9, output_file=None): 
        super().__init__()
        self.base_models = base_models
        self.epsilon = epsilon
        self.fading_factor = fading_factor

        # Store nb times a model has been selected 
        self.model_slection_freq = np.zeros(len(self.base_models))

        # Store estimates for each base model for selection 
        self.estimated_rewards = np.zeros(len(self.base_models))

        # Keep track of selected model at each step
        self.selected_model_idx = None

        # Store previous predictions
        # TODO: Change to  handle window of predictions and average measures
        self.previous_predictions = [None] *  len(self.base_models)

        # Store culumative errors incurred by each model
        self.cumulative_errors = np.zeros(len(self.base_models)) 

        self.first_run = True       
        self.exploit_vs_explore = []

        self.output_file = join(output_file, self.get_model_name() + '.csv')
        base_models_info = [b_model.get_info() for b_model in self.base_models]
        init_file(output_file=self.output_file, ensemble_size=len(self.base_models),base_models_info=base_models_info, global_info=self.get_info())
        

    def fit(self, X, y):
        raise NotImplementedError

    def partial_fit(self, X, y):
        # Update cumulative errors
        if not self.first_run:
            for i in range(len(self.previous_predictions)):
                self.cumulative_errors[i] = self.cumulative_errors[i]* self.fading_factor + (self.previous_predictions[i] - y)**2
                # Update rewards
                # TODO: Normalize or no ?
                self.estimated_rewards[i] = np.exp(- self.cumulative_errors[i])
                

        for b_model in self.base_models:
            b_model.partial_fit(X, y)
        self.first_run = False
            
    def predict(self, X):
        # Select model(arm)
        # This is epsilon-greeady approach
        # TODO: Explore other methods more complex
        rand_num = np.random.random()
        if self.epsilon > rand_num:
            # Explore
            selected_index = np.random.randint(len(self.base_models))
            self.exploit_vs_explore.append(1)
        else:
            # Exploit
            selected_index = np.argmax(self.estimated_rewards)
            self.exploit_vs_explore.append(0)
        # Update selection frequencies and selected model idx
        all_predictions = [b.predict(X)[0] for b in self.base_models]
        # Store all past prediction
        self.previous_predictions = all_predictions


        self.selected_model_idx = selected_index
        self.model_slection_freq[self.selected_model_idx] += 1

        # return prediction of the single selected model
        final_prediction = all_predictions[self.selected_model_idx]
        

        # Update base outputs File
        update_file(output_file=self.output_file, global_prediction=final_prediction, base_predictions=all_predictions, 
                    meta_predictions=[0]* len(self.base_models), base_selected_idx=[self.selected_model_idx])

        # Use prediction
        return [final_prediction]


    def predict_proba(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_info(self):
        info = '{}:'.format(type(self).__name__)
        info += ' - ensemble_size: {}'.format(len(self.base_models))
        info += ' - epsilon:{}'.format(self.epsilon)
        return info

    def get_model_name(self):
        return 'MAB_{}_SIZE_{}'.format(self.epsilon, len(self.base_models))

