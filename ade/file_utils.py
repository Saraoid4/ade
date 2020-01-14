import os
import ntpath

def init_file(output_file, ensemble_size, base_models_info, global_info ):

    if output_file is not None:
        # Check if path exists if not create it
        head, _ = ntpath.split(output_file)
        if not os.path.exists(head):
            os.makedirs(head)

        with open(output_file, 'w+') as f:
            f.write("# TEST CONFIGURATION BEGIN")

            for i in range(ensemble_size):
                if base_models_info is not None:
                    f.write("\n# [{}] {}".format(str(i), base_models_info[i]))
            f.write("\n# {}".format(global_info))
            f.write("\n# TEST CONFIGURATION END")
            header = '\nglobal_prediction'

            # Store all base-learners predictions
            for i in range(ensemble_size):
                header += ',base_prediction_[{0}]'.format(str(i))

            # store all meta-learners predictions
            for i in range(ensemble_size):
                header += ',meta_prediction_[{0}]'.format(str(i))

            # store for each base-learner if it was selected or no to predict on instance id
            for i in range(ensemble_size):
                header += ',base_selected_[{0}]'.format(str(i))


            f.write(header)

def update_file(output_file, global_prediction, base_predictions, meta_predictions, base_selected_idx):
    if output_file is not None:
        # iterate over multiple predictions
        # Must follow the exact same order as in _init_file
        new_line = '{:.6f}'.format(global_prediction)

        for i in range(len(base_predictions)):
            new_line += ',{:.6f}'.format(base_predictions[i])

        for i in range(len(meta_predictions)):
            new_line += ',{:.6f}'.format(meta_predictions[i])

        selectd_idx = [0] * len(base_predictions)

        for i in base_selected_idx:
            selectd_idx[i] = 1


        for i in range(len(selectd_idx)):
            new_line += ',{}'.format(selectd_idx[i])
        new_line = '\n' + new_line

        with open(output_file, 'a') as f:
            f.write(new_line)