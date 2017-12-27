from math import floor, ceil
from time import time
from pathlib import Path
from zipfile import ZipFile
from urllib.request import urlretrieve
from contextlib import contextmanager
import random

from pprint import pprint
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from keras.layers import Input, Embedding, Flatten, merge, Dense, Dropout
from keras.layers import BatchNormalization
from keras.models import Model
from dask import delayed, compute


DEFAULT_LOSS = 'cross_entropy'
ML_100K_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML_100K_FILENAME = Path(ML_100K_URL.rsplit('/', 1)[1])
ML_100K_FOLDER = Path('ml-100k')
RESULTS_FILENAME = 'results.json'
MODEL_FILENAME = 'model.h5'


if not ML_100K_FILENAME.exists():
    print('Downloading %s to %s...' % (ML_100K_URL, ML_100K_FILENAME))
    urlretrieve(ML_100K_URL, ML_100K_FILENAME.name)


if not ML_100K_FOLDER.exists():
    print('Extracting %s to %s...' % (ML_100K_FILENAME, ML_100K_FOLDER))
    ZipFile(ML_100K_FILENAME.name).extractall('.')


all_ratings = pd.read_csv(ML_100K_FOLDER / 'u.data', sep='\t',
                          names=["user_id", "item_id", "rating", "timestamp"])


DEFAULT_PARAMS = dict(
    embedding_size=16,
    hidden_size=64,
    n_hidden=4,
    dropout_embedding=0.3,
    dropout_hidden=0.3,
    use_batchnorm=True,
    loss=DEFAULT_LOSS,
    optimizer='adam',
    batch_size=64,
)


COMMON_SEARCH_SPACE = dict(
    embedding_size=[16, 32, 64, 128],
    dropout_embedding=[0, 0.2, 0.5],
    dropout_hidden=[0, 0.2, 0.5],
    use_batchnorm=[True, False],
    loss=['mse', 'mae', 'cross_entropy'],
    batch_size=[16, 32, 64, 128],
)

SEARCH_SPACE = [
    dict(n_hidden=[0], **COMMON_SEARCH_SPACE),
    dict(n_hidden=[1, 2, 3, 4, 5],
         hidden_size=[32, 64, 128, 256, 512],
         **COMMON_SEARCH_SPACE),
]


def bootstrap_ci(func, data_args, ci_range=(0.025, 0.975), n_iter=10000,
                 random_state=0):
    rng = np.random.RandomState(random_state)
    n_samples = data_args[0].shape[0]
    results = []
    for i in range(n_iter):
        # sample n_samples out of n_samples with replacement
        idx = rng.randint(0, n_samples - 1, n_samples)
        resampled_args = [np.asarray(arg)[idx] for arg in data_args]
        results.append(func(*resampled_args))
    results = np.sort(results)
    return (results[floor(ci_range[0] * n_iter)],
            results[ceil(ci_range[1] * n_iter)])


def make_model(user_input_dim, item_input_dim,
               embedding_size=16, hidden_size=64, n_hidden=4,
               dropout_embedding=0.3, dropout_hidden=0.3,
               optimizer='adam', loss=DEFAULT_LOSS, use_batchnorm=True,
               **ignored_args):

    user_id_input = Input(shape=[1], name='user')
    item_id_input = Input(shape=[1], name='item')

    user_embedding = Embedding(output_dim=embedding_size,
                               input_dim=user_input_dim,
                               input_length=1,
                               name='user_embedding')(user_id_input)
    item_embedding = Embedding(output_dim=embedding_size,
                               input_dim=item_input_dim,
                               input_length=1,
                               name='item_embedding')(item_id_input)

    user_vecs = Flatten()(user_embedding)
    item_vecs = Flatten()(item_embedding)

    input_vecs = merge([user_vecs, item_vecs], mode='concat')
    x = Dropout(dropout_embedding)(input_vecs)

    for i in range(n_hidden):
        x = Dense(hidden_size, activation='relu')(x)
        if i < n_hidden - 1:
            x = Dropout(dropout_hidden)(x)
            if use_batchnorm:
                x = BatchNormalization()(x)

    if loss == 'cross_entropy':
        y = Dense(output_dim=5, activation='softmax')(x)
        model = Model(input=[user_id_input, item_id_input], output=y)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    else:
        y = Dense(output_dim=1)(x)
        model = Model(input=[user_id_input, item_id_input], output=y)
        model.compile(optimizer='adam', loss=loss)
    return model


@contextmanager
def transactional_open(path, mode='wb'):
    tmp_path = path.with_name(path.name + '.tmp')
    with tmp_path.open(mode=mode) as f:
        yield f
    tmp_path.rename(path)


@contextmanager
def transactional_fname(path):
    tmp_path = path.with_name(path.name + '.tmp')
    yield str(tmp_path)
    tmp_path.rename(path)


def _compute_scores(model, prefix, user_id, item_id, rating, loss):
    preds = model.predict([user_id, item_id])
    preds = preds.argmax(axis=1) + 1 if loss == 'cross_entropy' else preds
    mse = mean_squared_error(preds, rating)
    mae = mean_absolute_error(preds, rating)
    mae_ci_min, mae_ci_max = bootstrap_ci(mean_absolute_error, [preds, rating])
    results = {}
    results[prefix + '_mse'] = mse
    results[prefix + '_mae'] = mae
    results[prefix + '_mae_ci_min'] = mae_ci_min
    results[prefix + '_mae_ci_max'] = mae_ci_max
    return results, preds


def evaluate_one(**kwargs):
    # Create a single threaded TF session for this Python thread:
    # parallelism is leveraged at a coarser level with dask
    session = tf.Session(
        # graph=tf.Graph(),
        config=tf.ConfigProto(intra_op_parallelism_threads=1))

    with session.as_default():
        # graph-level deterministic weights init
        tf.set_random_seed(0)
        _evaluate_one(**kwargs)


def _evaluate_one(**kwargs):
    params = DEFAULT_PARAMS.copy()
    params.update(kwargs)
    params_digest = joblib.hash(params)

    results = params.copy()
    results['digest'] = params_digest
    results_folder = Path('results')
    results_folder.mkdir(exist_ok=True)
    folder = results_folder.joinpath(params_digest)
    folder.mkdir(exist_ok=True)
    if len(list(folder.glob("*/results.json"))) == 4:
        print('Skipping')

    split_idx = params.get('split_idx', 0)
    print("Evaluating model on split #%d:" % split_idx)
    pprint(params)

    ratings_train, ratings_test = train_test_split(
        all_ratings, test_size=0.2, random_state=split_idx)
    max_user_id = all_ratings['user_id'].max()
    max_item_id = all_ratings['item_id'].max()

    user_id_train = ratings_train['user_id']
    item_id_train = ratings_train['item_id']
    rating_train = ratings_train['rating']

    user_id_test = ratings_test['user_id']
    item_id_test = ratings_test['item_id']
    rating_test = ratings_test['rating']

    loss = params.get('loss', DEFAULT_LOSS)
    if loss == 'cross_entropy':
        target_train = rating_train - 1
    else:
        target_train = rating_train

    model = make_model(max_user_id + 1, max_item_id + 1, **params)
    results['model_size'] = sum(w.size for w in model.get_weights())
    nb_epoch = 5
    epochs = 0
    for i in range(4):
        epochs += nb_epoch
        t0 = time()
        model.fit([user_id_train, item_id_train], target_train,
                  batch_size=params['batch_size'],
                  nb_epoch=nb_epoch, shuffle=True, verbose=False)
        epoch_duration = (time() - t0) / nb_epoch
        train_scores, train_preds = _compute_scores(
            model, 'train', user_id_train, item_id_train, rating_train, loss)
        results.update(train_scores)
        test_scores, test_preds = _compute_scores(
            model, 'test', user_id_test, item_id_test, rating_test, loss)
        results.update(test_scores)

        results['epoch_duration'] = epoch_duration
        results['epochs'] = epochs

        subfolder = folder.joinpath("%03d" % epochs)
        subfolder.mkdir(exist_ok=True)

        # Transactional results saving to avoid file corruption on ctrl-c
        results_filepath = subfolder.joinpath(RESULTS_FILENAME)
        with transactional_open(results_filepath, mode='w') as f:
            json.dump(results, f)

        model_filepath = subfolder.joinpath(MODEL_FILENAME)
        with transactional_fname(model_filepath) as fname:
            model.save(fname)

        # Save predictions and true labels to be able to recompute new scores
        # later
        with transactional_open(subfolder / 'test_preds.npy', mode='wb') as f:
            np.save(f, test_preds)
        with transactional_open(subfolder / 'train_preds.npy', mode='wb') as f:
            np.save(f, test_preds)
        with transactional_open(subfolder / 'ratings.npy', mode='wb') as f:
            np.save(f, rating_test)

    return params_digest


def _model_complexity_proxy(params):
    # Quick approximation of the number of tunable parameter to rank models
    # by increasing complexity
    embedding_size = params['embedding_size']
    n_hidden = params['n_hidden']
    if n_hidden == 0:
        return embedding_size * 2
    else:
        hidden_size = params['hidden_size']
        return (2 * embedding_size * hidden_size +
                (n_hidden - 1) * hidden_size ** 2)


if __name__ == "__main__":
    seed = 0
    n_params = 500
    all_combinations = list(ParameterGrid(SEARCH_SPACE))
    random.Random(seed).shuffle(all_combinations)
    sampled_params = all_combinations[:n_params]
    sampled_params.sort(key=_model_complexity_proxy)
    evaluations = []
    for params in sampled_params:
        for split_idx in range(3):
            evaluations.append(delayed(evaluate_one)(
                split_idx=split_idx, **params))
    compute(*evaluations)
