import timesynth as ts
import numpy as np
import pandas as pd
import random
import itertools
import matplotlib.pyplot as plt

# AR models import
class TimeSeriesData:
    def __init__(self, num_datapoints, test_size=0.2, max_t=20, num_prev=1,
                 noise_var=1, mean=0, std=1):
        """
        Template class for generating time series data.
        :param test_size: in (0,1), data to be used in test set as a fraction of all data generated.
        """
        self.num_datapoints = num_datapoints
        self.test_size = test_size
        self.num_prev = num_prev
        self.max_t = max_t
        self.data = None
        self.noise_var = noise_var

        # Normal dist parameters
        self.mean = mean
        self.std = std

        self.y = np.zeros(num_datapoints + num_prev*4) # TODO: check this
        self.bayes_preds = np.copy(self.y)

        # Generate data and reshape data
        self.create_data()

        # Split into training and test sets
        self.train_test_split()

    def create_data(self):
        self.generate_data()
        self.reshape_data()

    def generate_data(self):
        """Generates data in self.y, may take as implicit input timesteps self.t.
        May also generate Bayes predictions."""
        raise NotImplementedError("Generate data method not implemented.")

    def reshape_data(self):
        self.x = np.reshape([self.y[i:i + self.num_prev] for i in range(
            self.num_datapoints)], (-1, self.num_prev))
        self.y = np.copy(self.y[self.num_prev:])
        self.bayes_preds = np.copy(self.bayes_preds[self.num_prev:])

    def train_test_split(self):
        test_size = int(len(self.y) * self.test_size)
        self.data = [self.X_train, self.X_test, self.y_train,
                     self.y_test] = \
                    self.x[:-test_size], self.x[-test_size:], \
                    self.y[:-test_size], self.y[-test_size:]
        self.bayes_preds = [self.bayes_train_preds, self.bayes_test_preds] = self.bayes_preds[:-test_size], self.bayes_preds[-test_size:]

    def return_data(self):
        return self.data

    def return_train_test(self):
        return self.X_train, self.y_train, self.X_test, self.y_test


class ARData(TimeSeriesData):
    """Class to generate autoregressive data."""

    def __init__(self, *args, coeffs=None, **kwargs):
        self.given_coeffs = coeffs
        super(ARData, self).__init__(*args, **kwargs)

        if coeffs is not None:
            self.num_prev = len(coeffs) - 1

    def generate_data(self):
        self.generate_coefficients()
        self.generate_initial_points()

        # + 3*self.num_prev because we want to cut first (3*self.num_prev) datapoints later
        # so dist is more stationary (else initial num_prev datapoints will stand out as diff dist)
        for i in range(self.num_datapoints + 3 * self.num_prev):
            # Generate y value if there was no noise
            # (equivalent to Bayes predictions: predictions from oracle that knows true parameters (coefficients))
            self.bayes_preds[i + self.num_prev] = np.dot(self.y[i:self.num_prev + i][::-1], self.coeffs)
            # Add noise
            self.y[i + self.num_prev] = self.bayes_preds[i + self.num_prev] + self.noise()

        # Cut first 20 points so dist is roughly stationary
        self.bayes_preds = self.bayes_preds[3 * self.num_prev:]
        self.y = self.y[3 * self.num_prev:]

    def generate_coefficients(self):
        if self.given_coeffs is not None:
            self.coeffs = self.given_coeffs
        else:
            filter_stable = False
            # Keep generating coefficients until we come across a set of coefficients
            # that correspond to stable poles
            while not filter_stable:
                true_theta = np.random.random(self.num_prev) - 0.5
                coefficients = np.append(1, -true_theta)
                # check if magnitude of all poles is less than one
                if np.max(np.abs(np.roots(coefficients))) < 1:
                    filter_stable = True
            self.coeffs = true_theta

    def generate_initial_points(self):
        # Initial datapoints distributed as N(0,1)
        # self.y[:self.num_prev] = np.random.randn(self.num_prev)
        self.y[:self.num_prev] = np.random.normal(loc=self.mean, scale=self.std,  size=self.num_prev)

    def noise(self):
        # Noise distributed as N(0, self.noise_var)
        return self.noise_var * np.random.randn()

def generate_ar_time_series(concept_length, nb_unique_concepts, nb_recurrent=0, shuffle=False, ar_order = None, noise=False):
    # Initializing TimeSampler
    time_sampler = ts.TimeSampler(stop_time=2000)
    # Sampling regular time samples
    regular_time_samples = time_sampler.sample_regular_time(num_points=concept_length)


    if ar_order is None:
    # No order specified, should be random and different for each concept
    # Generate nb_unique_concepts different random p values for the order
        ar_order_values = np.random.randint(low=1, high=5, size=nb_unique_concepts)
        means = np.random.randint(low=10, high=100*nb_unique_concepts, size=nb_unique_concepts)
        stds = np.random.randint(low=1, high=100*nb_unique_concepts, size=nb_unique_concepts)

    else:
        ar_order_values = [ar_order]*nb_unique_concepts

    print('ar_order_values', ar_order_values)

    concepts_samples = []
    ar_params_list=[]

    for c in range(nb_unique_concepts):
        # print(c)
        ar_order = ar_order_values[c]
        # Generate different concepts
        # Generate different ar_params
        # [ 2.93167412 -0.40976317  2.38339705  2.05085328] order 4

        filter_stable = False
        # Keep generating coefficients until we come across a set of coefficients
        # that correspond to stable poles

        while not filter_stable:
            # true_theta = np.random.random(self.num_prev) - 0.5
            true_theta = np.random.uniform(low=1, high=5, size=ar_order) -0.5
            coefficients = np.append(1, -true_theta)

            # check if magnitude of all poles is less than one
            if np.max(np.abs(np.roots(coefficients))) < 1:
                filter_stable = True

        ar_params = true_theta

        start_value = np.random.normal(loc=means[c], scale=stds[c], size=len(ar_params))
        print('ar_params', ar_params)
        ar_p = ts.signals.AutoRegressive(ar_param=list(ar_params), start_value=start_value)

        ar_p_series = ts.TimeSeries(signal_generator=ar_p)
        samples, signals, errors = ar_p_series.sample(regular_time_samples)
        concepts_samples.append(samples)


    # Generate recurrent concepts
    if nb_recurrent >=2:
        concepts_samples = concepts_samples * nb_recurrent

    # Re-order concepts if needed
    if shuffle:
        random.shuffle(concepts_samples)

    return list(itertools.chain.from_iterable(concepts_samples))

def generate_car_time_series(concept_length, nb_unique_concepts, nb_recurrent=0, shuffle=False, noise=False):
    # Initializing TimeSampler
    time_sampler = ts.TimeSampler(stop_time=1000)
    # Sampling regular time samples
    regular_time_samples = time_sampler.sample_regular_time(num_points=concept_length)


    concepts_samples = []
    start_value = 0
    car_params_list = np.random.uniform(low=0, high=1, size=nb_unique_concepts)

    for c in range(nb_unique_concepts):
        # Generate different concepts with different car params


        car = ts.signals.CAR(ar_param=car_params_list[c], start_value=start_value)

        car_series = ts.TimeSeries(signal_generator=car)
        samples, signals, errors = car_series.sample(regular_time_samples)
        concepts_samples.append(samples)
        start_value= samples[-1]

    # Generate recurrent concepts
    if nb_recurrent >=2:
        concepts_samples = concepts_samples * nb_recurrent

    # Re-order concepts if needed
    if shuffle:
        random.shuffle(concepts_samples)

    return list(itertools.chain.from_iterable(concepts_samples))


def generate_synth_data(concept_length, nb_unique_concepts, nb_recurrent, shuffle=False, ar_order=None):
    all_samples = []
    # Generate different orders
    if ar_order is None:
        # No order specified, should be random and different for each concept
        # Generate nb_unique_concepts different random p values for the order
        ar_order_values = np.random.randint(low=1, high=10, size=nb_unique_concepts)
        means = np.random.randint(low=10, high=100, size=nb_unique_concepts)
        stds = np.random.randint(low=10, high=100, size=nb_unique_concepts)
    else:
        ar_order_values = [ar_order] * nb_unique_concepts

    print('ar_order_values', ar_order_values)
    print('means', means)
    print('stds', stds)


    for c in range(nb_unique_concepts):
        # Generate a unique and different concept
        num_prev = ar_order_values[c]
        mean = means[c]
        std = stds[c]
        stable_ar = ARData(num_datapoints=concept_length, num_prev=num_prev, mean=mean, std=std)
        print(stable_ar.coeffs)
        all_samples.append(stable_ar.y)

    # Generate recurrent concepts
    if nb_recurrent >= 2:
        all_samples = all_samples * nb_recurrent

    # Re-order concepts if needed
    if shuffle:
        random.shuffle(all_samples)

    return list(itertools.chain.from_iterable(all_samples))

def get_noise(noise_type, noise_mean, noise_std, tau, start_value=None):

    if noise_type == 'white':
        # If white gaussian noise
        noise = ts.noise.GaussianNoise(mean=noise_mean, std=noise_std)
    elif noise_type == 'red':
        #If red noise
        noise = ts.noise.RedNoise(mean=noise_mean, std=noise_std, tau=tau, start_value=start_value)
    else:
        raise ValueError('Not Valid Noise Type', noise_type)

    return noise

def generate_harmonic_signal(concept_length, amplitude=1, frequency=0.25,noise_type='white', noise_mean=0,
                             noise_std=0.3, tau=None, noise_start_value=0):

    time_sampler = ts.TimeSampler(stop_time=20)
    # Sampling irregular time samples
    # TODO: Change the hardcoded num points
    irregular_time_samples = time_sampler.sample_irregular_time(num_points=concept_length*2, keep_percentage=50)

    # Initializing Sinusoidal signal
    sinusoid = ts.signals.Sinusoidal(amplitude=amplitude, frequency=frequency)

    # Injecting gaussian noise
    # Initializing  noise
    # TODO: set up start value for noise
    noise = get_noise(noise_type=noise_type, noise_mean=noise_mean, noise_std=noise_std, tau=tau, start_value=noise_start_value)



    # Initializing TimeSeries class with the signal and noise objects
    timeseries = ts.TimeSeries(sinusoid, noise_generator=noise)

    # Sampling using the irregular time samples
    samples, signals, errors = timeseries.sample(irregular_time_samples)

    return samples

def generate_pseudo_periodic_signal(concept_length, amplitude=1, frequency=1, ampSD=0.1, freqSD=0.1,
                               noise_type='white', noise_mean=0, noise_std=1, tau=0.8, noise_start_value=0):
    # Initializing TimeSampler
    time_sampler_pp = ts.TimeSampler(stop_time=20)
    # Sampling irregular time samples
    irregular_time_samples = time_sampler_pp.sample_irregular_time(num_points=concept_length*2, keep_percentage=50)

    # Initializing Pseudoperiodic signal
    pseudo_periodic = ts.signals.PseudoPeriodic(amplitude=amplitude,frequency=frequency, freqSD=freqSD, ampSD=ampSD)

    # Initialize noise
    noise = get_noise(noise_type=noise_type, noise_mean=noise_mean, noise_std=noise_std, tau=tau,
                      start_value=noise_start_value)

    #Initialize time series
    timeseries = ts.TimeSeries(pseudo_periodic, noise_generator=noise)
    # Sampling using the irregular time samples
    samples, signals, errors = timeseries.sample(irregular_time_samples)

    return samples


def generate_gaussian_signal(concept_length, kernel="SE", lengthscale=1., mean=0., variance=1., c=1., gamma=1., alpha=1.,
                             offset=0., nu=5./2, p=1.,
                             noise_type='white', noise_mean=0, noise_std=1, tau=0.8, noise_start_value=0):

    # Initializing TimeSampler
    time_sampler_pp = ts.TimeSampler(stop_time=20)
    # Sampling irregular time samples
    irregular_time_samples = time_sampler_pp.sample_irregular_time(num_points=concept_length * 2, keep_percentage=50)

    #Initialize gaussian process
    gaussian_process = ts.signals.GaussianProcess(kernel=kernel, lengthscale=lengthscale, mean=mean, variance=variance,
                                                  c=c, gamma=gamma, alpha=alpha, nu=nu, p=p)

    # Get noise
    noise = get_noise(noise_type=noise_type, noise_mean=noise_mean, noise_std=noise_std, tau=tau,
                      start_value=noise_start_value)

    # Initialize time series
    timeseries = ts.TimeSeries(gaussian_process, noise_generator=noise)


    # Sampling using the irregular time samples
    samples, signals, errors = timeseries.sample(irregular_time_samples)

    return samples

def generate_ar_signal(concept_length, ar_params, std=1, start_values=[None], noise_type='white',
                       noise_mean=0, noise_std=1, tau=0.8, noise_start_value=0):
    if start_values != [None]:
        if len(ar_params) != len(start_values):
            raise Exception('Order and AR parameters must have same length')
    '''
    else:
         if any(l != order for l in [len(ar_parameters),len(start_values)]):
             raise Exception('Order, AR params and start values must be same length')'''

    # Initializing TimeSampler
    time_sampler = ts.TimeSampler(stop_time=20)
    # Sampling regular time samples
    regular_time_samples = time_sampler.sample_regular_time(num_points=concept_length)

    # Initializing AR(len(ar_parameters)) model
    ar_p = ts.signals.AutoRegressive(ar_param=ar_params)

    # Get noise
    noise = get_noise(noise_type=noise_type, noise_mean=noise_mean, noise_std=noise_std, tau=tau,
                      start_value=noise_start_value)

    # Initialize time series
    ar_p_series = ts.TimeSeries(signal_generator=ar_p, noise_generator=noise)

    samples, signals, errors = ar_p_series.sample(regular_time_samples)


    return samples

def generate_concepts_chain(concepts_settings, nb_repeat=None, shuffle=False):
    concepts_samples = []
    for setting in concepts_settings:
        if setting['signal_type'] == 'Gaussian':
            samples = generate_gaussian_signal(**setting['signal_parameters'])
        elif setting['signal_type'] == 'PseudoPeriodic':
            samples = generate_pseudo_periodic_signal(**setting['signal_parameters'])
        elif setting['signal_type'] == 'Harmonic':
            samples = generate_harmonic_signal(**setting['signal_parameters'])
        elif setting['signal_type'] == 'AR':
            samples = generate_ar_signal(**setting['signal_parameters'])
        else:
            raise ValueError('Invalid Signal Type', concepts_settings['signal_type'])

        concepts_samples.append(samples)


    # Generate recurrent concepts
    if nb_repeat >= 2:
        concepts_samples = concepts_samples * nb_repeat

        # Re-order concepts if needed
    if shuffle:
        random.shuffle(concepts_samples)

    return list(itertools.chain.from_iterable(concepts_samples))


def generate_series(key_model, value_model, dataset_dirname, data_output_dir, data_name):
    pass


if __name__ == '__main__':
    signal_types_values = {'Gaussian':{'kernel':['SE', 'Exponential', 'Constant', 'RQ', 'Linear', 'Matern', 'Periodic'], 
                                       'mean':[], 
                                       'variance':[] },
                           'PseudoPeriodic':{'amplitude':[], 'frequency':[], 'ampSD':[], 'freqSD':[]}, 
                           'Harmonic': {'amplitude':[], 'frequency':[], 'ftype': []}}

    

    concepts_chain = [
        {'signal_type': 'Gaussian', 'signal_parameters': {'concept_length': 2000, 'kernel': 'SE', 'mean':300, 'variance':300}},
        {'signal_type': 'PseudoPeriodic', 'signal_parameters': {'concept_length': 2000, 'amplitude':200, 'frequency':0.25}},
        {'signal_type': 'Harmonic', 'signal_parameters': {'concept_length': 2000, 'amplitude':50}},
        {'signal_type': 'Gaussian', 'signal_parameters': {'concept_length': 2000, 'kernel': 'Linear', 'mean':300, 'variance':300}},
        # {'signal_type': 'Gaussian', 'signal_parameters': {'concept_length': 1000, 'kernel': 'Constant', 'mean':0, 'variance':1}},
        {'signal_type': 'Gaussian', 'signal_parameters': {'concept_length': 2000, 'kernel': 'Exponential', 'mean':300, 'variance':300}},
        {'signal_type': 'Gaussian', 'signal_parameters': {'concept_length': 2000, 'kernel': 'Periodic', 'mean':300, 'variance':300}},
        # {'signal_type': 'Gaussian', 'signal_parameters': {'concept_length': 1000, 'kernel': 'Matern', 'mean':300, 'variance':300}},
        {'signal_type': 'Gaussian', 'signal_parameters': {'concept_length': 2000, 'kernel': 'RQ', 'mean':300, 'variance':300}},
        
    ]

    nb_time_series = 5
    for j in range(nb_time_series):
        global_concepts = []
        nb_random_repeats = 2
        concept_length = random.randint(1, 10) * 100 

        for i in range(nb_random_repeats):
            for concept_setting in concepts_chain:

                noise_setting = {'noise_type':'white', 'noise_mean': random.uniform(0,1), 'noise_std':random.uniform(0, 1), 
                                'tau':random.uniform(0, 1), 'noise_start_value': random.uniform(0, 1)}
                
                concept_setting['signal_parameters'].update(noise_setting)

                if 'mean' in concept_setting['signal_parameters'].keys():
                    concept_setting['signal_parameters']['mean'] = random.randint(10, 500)
                
                if 'variance' in concept_setting['signal_parameters'].keys():
                    concept_setting['signal_parameters']['variance'] = random.randint(10, 500)
                
                if 'amplitude' in concept_setting['signal_parameters'].keys():
                    concept_setting['signal_parameters']['amplitude'] = random.randint(10, 500)
                
                if 'frequency' in concept_setting['signal_parameters'].keys():
                    concept_setting['signal_parameters']['frequency'] = random.uniform(0, 1)
                
                concept_setting['signal_parameters']['concept_length'] = concept_length

            global_concepts.extend(concepts_chain)
            
        samples = generate_concepts_chain(concepts_settings=global_concepts, nb_repeat=3, shuffle=True)
        df = pd.DataFrame({'target': samples})
        df.to_csv('synth_date_{}_lenght_{}.csv'.format(j, concept_length), index=False)
        plt.plot(list(range(1, len(samples)+1)), samples)
        plt.show()


