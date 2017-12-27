import matplotlib.pyplot as plt

from pathlib import Path
from zipfile import ZipFile
from urllib.request import urlretrieve

from keras.layers import Input, Embedding, Flatten, merge, Dense, Dropout, Lambda
from keras.models import Model
import keras.backend as K
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

ML_1M_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML_1M_FILENAME = Path(ML_1M_URL.rsplit('/', 1)[1])
ML_1M_FOLDER = Path('ml-1m')

if not ML_1M_FILENAME.exists():
    print('Downloading %s to %s...' % (ML_1M_URL, ML_1M_FILENAME))
    urlretrieve(ML_1M_URL, ML_1M_FILENAME.name)

if not ML_1M_FOLDER.exists():
    print('Extracting %s to %s...' % (ML_1M_FILENAME, ML_1M_FOLDER))
    ZipFile(ML_1M_FILENAME.name).extractall('.')


all_ratings = pd.read_csv(ML_1M_FOLDER / 'ratings.dat', sep='::',
                          names=["user_id", "item_id", "rating", "timestamp"])
all_ratings.head()

max_user_id = all_ratings['user_id'].max()
max_item_id = all_ratings['item_id'].max()

from sklearn.model_selection import train_test_split

ratings_train, ratings_test = train_test_split(
    all_ratings, test_size=0.2, random_state=0)

user_id_train = ratings_train['user_id']
item_id_train = ratings_train['item_id']
rating_train = ratings_train['rating']

user_id_test = ratings_test['user_id']
item_id_test = ratings_test['item_id']
rating_test = ratings_test['rating']

user_id_input = Input(shape=[1], name='user')
item_id_input = Input(shape=[1], name='item')

embedding_size = 32
user_embedding = Embedding(output_dim=embedding_size, input_dim=max_user_id + 1,
                           input_length=1, name='user_embedding')(user_id_input)
item_embedding = Embedding(output_dim=embedding_size, input_dim=max_item_id + 1,
                           input_length=1, name='item_embedding')(item_id_input)

# reshape from shape: (batch_size, input_length, embedding_size)
# to shape: (batch_size, input_length * embedding_size) which is
# equal to shape: (batch_size, embedding_size)
user_vecs = Flatten()(user_embedding)
item_vecs = Flatten()(item_embedding)

input_vecs = merge([user_vecs, item_vecs], mode='concat')
input_vecs = Dropout(0.5)(input_vecs)

dense_size = 64
x = Dense(dense_size, activation='relu')(input_vecs)
x = Dense(dense_size, activation='relu')(input_vecs)
y = Dense(output_dim=1)(x)

model = Model(input=[user_id_input, item_id_input], output=y)
model.compile(optimizer='adam', loss='mae')

initial_train_preds = model.predict([user_id_train, item_id_train])

history = model.fit([user_id_train, item_id_train], rating_train,
                    batch_size=64, nb_epoch=15, validation_split=0.1,
                    shuffle=True)

test_preds = model.predict([user_id_test, item_id_test])
print("Final test MSE: %0.3f" % mean_squared_error(test_preds, rating_test))
print("Final test MAE: %0.3f" % mean_absolute_error(test_preds, rating_test))
