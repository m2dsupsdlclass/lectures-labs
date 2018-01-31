# display figures in the notebook
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical


m = joblib.Memory(cachedir='/tmp')


@m.cache()
def make_curves(random_state=42):
    digits = load_digits()
    rng = np.random.RandomState(random_state)

    data = np.asarray(digits.data, dtype='float32')
    target = np.asarray(digits.target, dtype='int32')

    # Add noise in the labels to cause more overfitting
    target[:200] = rng.randint(0, 10, size=200)

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.15, random_state=random_state)

    # mean = 0 ; standard deviation = 1.0
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    Y_train = to_categorical(y_train)

    N = X_train.shape[1]
    H = 1024
    K = 10

    curve_data = {}
    for dropout in [0, 0.2, 0.8]:
        tf.set_random_seed(random_state)
        model = Sequential()
        model.add(Dense(H, input_dim=N, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(K, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(K, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(K))
        model.add(Activation("softmax"))

        model.compile(optimizer=SGD(lr=0.1), loss='categorical_crossentropy')

        history = model.fit(X_train, Y_train, batch_size=128, nb_epoch=150,
                            validation_split=0.5, shuffle=True)
        curve_data[(dropout, 'train')] = history.history['loss']
        curve_data[(dropout, 'validation')] = history.history['val_loss']
    return curve_data


# Compute learning curves
curve_data = make_curves()


# Only without dropout
plt.figure()
for (dropout, loss_type), values in sorted(curve_data.items()):
    if dropout > 0:
        continue
    label = loss_type
    linestyle = "--" if loss_type == 'validation' else '-'
    color = "#1f77b4"
    label += ", no dropout"

    plt.plot(values, linestyle=linestyle, c=color, label=label)

plt.legend(loc='best')
plt.ylim(0, 2.4)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("MLP with 3 hidden layers and noisy labels")
plt.savefig('dropout_curves_1.svg')

# Both curves, weak dropout
plt.figure()
for (dropout, loss_type), values in sorted(curve_data.items()):
    if dropout > 0.2:
        continue
    label = loss_type
    linestyle = "--" if loss_type == 'validation' else '-'
    if dropout == 0:
        color = "#1f77b4"
        label += ", no dropout"
    else:
        color = "#ff7f0e"
        label += ", dropout p=%0.1f" % dropout

    plt.plot(values, linestyle=linestyle, c=color, label=label)

plt.legend(loc='best')
plt.ylim(0, 2.4)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("MLP with 3 hidden layers and noisy labels")
plt.savefig('dropout_curves_2.svg')


# Both curves, strong dropout
plt.figure()
for (dropout, loss_type), values in sorted(curve_data.items()):
    if dropout == 0.2:
        continue
    label = loss_type
    linestyle = "--" if loss_type == 'validation' else '-'
    if dropout == 0:
        color = "#1f77b4"
        label += ", no dropout"
    else:
        color = "#ff7f0e"
        label += ", dropout p=%0.1f" % dropout

    plt.plot(values, linestyle=linestyle, c=color, label=label)

plt.legend(loc='best')
plt.ylim(0, 2.4)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("MLP with 3 hidden layers and noisy labels")
plt.savefig('dropout_curves_3.svg')
