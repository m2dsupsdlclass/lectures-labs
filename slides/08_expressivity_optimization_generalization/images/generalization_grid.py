import json
import os
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network.multilayer_perceptron import ACTIVATIONS
from sklearn.model_selection import ParameterGrid
from sklearn.utils import shuffle
from sklearn.utils.extmath import safe_sparse_dot

import joblib


model_filename = 'models.log'
evaluations_filename = 'evaluations.log'



def logits(m, X):
    sigma = ACTIVATIONS[m.activation]
    a = X
    for i in range(m.n_layers_ - 1):
        a = safe_sparse_dot(a, m.coefs_[i])
        a += m.intercepts_[i]
        if (i + 1) != (m.n_layers_ - 1):
            activations = sigma(a)
    return a


def lipschitz(m):
    return np.prod([max(np.linalg.svd(w)[1]) for w in m.coefs_])


def margins(m, X, y):
    preds = logits(m, X).ravel()
#     correct_mask = (preds >= 0) == y
#     return np.abs(preds * correct_mask)
    return np.abs(preds)


def normalized_margins(m, X, y):
    return margins(m, X, y) / lipschitz(m)


def bartlett_complexity_mean(m, X, y):
    return 1 / normalized_margins(m, X, y).mean()


def bartlett_complexity_median(m, X, y):
    median = np.median(normalized_margins(m, X, y))
    if median == 0:
        return 0
    return 1 / median


def make_noisy_problem(n_samples_train=30, label_noise_rate=0.1, input_noise=0.15,
                       n_samples_test=3000, seed=0):
    rng = np.random.RandomState(seed)
    rng = np.random.RandomState(1)
    scaler = StandardScaler()

    X_train, y_train = make_moons(n_samples=n_samples_train, shuffle=True,
                                  noise=input_noise, random_state=rng)
    X_test, y_test = make_moons(n_samples=n_samples_test, shuffle=True,
                                noise=input_noise, random_state=rng)
    
    if label_noise_rate > 0:
        rnd_levels = rng.uniform(low=0., high=1., size=n_samples_train)
        noise_mask = rnd_levels <= label_noise_rate
        y_train[noise_mask] = rng.randint(low=0, high=2, size=noise_mask.sum())
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return (X_train, y_train), (X_test, y_test)


hidden_layer_sizes_range = [
    [16],      [64],     [256],     [512],     [1024],
    [16]  * 2, [64] * 2, [256] * 2, [512] * 2,
                         [256] * 3,
                         [256] * 4,
                         [256] * 5,
]

param_grid = [
    {
        'solver': ['sgd', 'adam'],
        'hidden_layer_sizes': hidden_layer_sizes_range,
        'activation': ['relu'],
        'random_state': [0],
        'learning_rate_init': [0.1, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [5000],
    },
    {
        'solver': ['lbfgs'],
        'hidden_layer_sizes': hidden_layer_sizes_range,
        'activation': ['relu'],
        'random_state': [0],
    },
]

if __name__ == '__main__':
    model_params = list(ParameterGrid(param_grid))
    with open(model_filename, 'w') as f:
        for params in model_params:
            model_id = joblib.hash(params)
            model_record = params.copy()
            model_record['model_id'] = model_id
            model_record['depth'] = len(params['hidden_layer_sizes'])
            model_record['width'] = max(params['hidden_layer_sizes'])
            f.write(json.dumps(model_record) + '\n')
            f.flush()

    model_params = shuffle(model_params, random_state=0)
    with open(evaluations_filename, 'w') as f:
        for n_samples_train in [30]:
            for label_noise_rate in np.linspace(0, 1, 11):
                print(f'\nn_samples: {n_samples_train}, label noise: {label_noise_rate:0.1f}')
                for data_seed in [0, 1]:
                    (X_train, y_train), (X_test, y_test) = make_noisy_problem(
                        n_samples_train, label_noise_rate, seed=data_seed)
                    for params in model_params:
                        model_id = joblib.hash(params)
                        m = MLPClassifier(**params).fit(X_train, y_train)
                        train_acc = m.score(X_train, y_train)
                        test_acc = m.score(X_test, y_test)
                        excess_risk = max(train_acc - test_acc, 0)
                        n_params = sum([np.product(w.shape) for w in m.coefs_])
                        n_params += sum([np.product(b.shape) for b in m.intercepts_])
                        evaluation_record = {
                            'model_id': model_id,
                            'n_samples_train': n_samples_train,
                            'label_noise_rate': label_noise_rate,
                            'train_acc': train_acc,
                            'test_acc': test_acc,
                            'excess_risk': excess_risk,
                            'lipschitz': lipschitz(m),
                            'mean_margins': margins(m, X_train, y_train).mean(),
                            'median_margins': np.median(margins(m, X_train, y_train)),
                            'bartlett_complexity_mean': bartlett_complexity_mean(m, X_train, y_train),
                            'bartlett_complexity_median': bartlett_complexity_median(m, X_train, y_train),
                            'mean_margins_test': margins(m, X_test, y_test).mean(),
                            'median_margins_test': np.median(margins(m, X_test, y_test)),
                            'bartlett_complexity_mean_test': bartlett_complexity_mean(m, X_test, y_test),
                            'bartlett_complexity_median_test': bartlett_complexity_median(m, X_test, y_test),
                            'n_params': int(n_params),
                        }
                        f.write(json.dumps(evaluation_record) + '\n')
                        f.flush()
                        print('.', end='', flush=True)