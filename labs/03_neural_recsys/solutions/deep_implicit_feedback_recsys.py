from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.layers import Concatenate, Dropout
from tensorflow.keras.regularizers import l2


class MLP(layers.Layer):
    def __init__(self, n_hidden=1, hidden_size=64, dropout=0.,
                 l2_reg=None):
        super().__init__()

        self.layers = [Dropout(dropout)]

        for _ in range(n_hidden):
            self.layers.append(Dense(hidden_size, activation="relu",
                                     kernel_regularizer=l2_reg))
            self.layers.append(Dropout(dropout))

        self.layers.append(Dense(1, activation="relu",
                                 kernel_regularizer=l2_reg))

    def call(self, x, training=False):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x


class DeepTripletModel(Model):
    def __init__(self, n_users, n_items, user_dim=32, item_dim=64, margin=1.,
                 n_hidden=1, hidden_size=64, dropout=0, l2_reg=None):
        super().__init__()

        l2_reg = None if l2_reg == 0 else l2(l2_reg)

        self.user_layer = Embedding(n_users, user_dim,
                                    input_length=1,
                                    input_shape=(1,),
                                    name='user_embedding',
                                    embeddings_regularizer=l2_reg)
        self.item_layer = Embedding(n_items, item_dim,
                                    input_length=1,
                                    name="item_embedding",
                                    embeddings_regularizer=l2_reg)

        self.flatten = Flatten()
        self.concat = Concatenate()

        self.mlp = MLP(n_hidden, hidden_size, dropout, l2_reg)

        self.margin_loss = MarginLoss(margin)

    def call(self, inputs, training=False):
        user_input = inputs[0]
        pos_item_input = inputs[1]
        neg_item_input = inputs[2]

        user_embedding = self.user_layer(user_input)
        user_embedding = self.flatten(user_embedding)
        pos_item_embedding = self.item_layer(pos_item_input)
        pos_item_embedding = self.flatten(pos_item_embedding)
        neg_item_embedding = self.item_layer(neg_item_input)
        neg_item_embedding = self.flatten(neg_item_embedding)

        # Similarity computation between embeddings
        pos_embeddings_pair = self.concat([user_embedding,
                                           pos_item_embedding])
        neg_embeddings_pair = self.concat([user_embedding,
                                           neg_item_embedding])

        pos_similarity = self.mlp(pos_embeddings_pair)
        neg_similarity = self.mlp(neg_embeddings_pair)

        return self.margin_loss([pos_similarity, neg_similarity])


class DeepMatchModel(Model):
    def __init__(self, user_layer, item_layer, mlp):
        super().__init__(name="MatchModel")

        self.user_layer = user_layer
        self.item_layer = item_layer
        self.mlp = mlp

        self.flatten = Flatten()
        self.concat = Concatenate()

    def call(self, inputs):
        user_input = inputs[0]
        pos_item_input = inputs[1]

        user_embedding = self.flatten(self.user_layer(user_input))
        pos_item_embedding = self.flatten(self.item_layer(pos_item_input))

        pos_embeddings_pair = self.concat([user_embedding, pos_item_embedding])
        pos_similarity = self.mlp(pos_embeddings_pair)

        return pos_similarity


hyper_parameters = dict(
    user_dim=32,
    item_dim=64,
    n_hidden=1,
    hidden_size=128,
    dropout=0.1,
    l2_reg=0.,
)
deep_triplet_model = DeepTripletModel(n_users, n_items,
                                      **hyper_parameters)
deep_match_model = DeepMatchModel(deep_triplet_model.user_layer,
                                  deep_triplet_model.item_layer,
                                  deep_triplet_model.mlp)

deep_triplet_model.compile(loss=identity_loss, optimizer='adam')
fake_y = np.ones_like(pos_data_train['user_id'])

n_epochs = 20

for i in range(n_epochs):
    # Sample new negatives to build different triplets at each epoch
    triplet_inputs = sample_triplets(pos_data_train, max_item_id,
                                     random_seed=i)

    # Fit the model incrementally by doing a single pass over the
    # sampled triplets.
    deep_triplet_model.fit(triplet_inputs, fake_y, shuffle=True,
                           batch_size=64, epochs=1)


# Monitor the convergence of the model
test_auc = average_roc_auc(
    deep_match_model, pos_data_train, pos_data_test)
print("Epoch %d/%d: test ROC AUC: %0.4f"
      % (i + 1, n_epochs, test_auc))