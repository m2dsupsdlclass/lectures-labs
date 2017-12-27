# Analysis:
#
# Both models have again exactly the same number of parameters,
# namely the parameters of the 2 embeddings:
#
# - user embedding: n_users x user_dim
# - item embedding: n_items x item_dim
#
# and the parameters of the MLP model used to compute the
# similarity score of an (user, item) pair:
#
# - first hidden layer weights: (user_dim + item_dim) * hidden_size
# - first hidden biases: hidden_size
# - extra hidden layers weights: hidden_size * hidden_size
# - extra hidden layers biases: hidden_size
# - output layer weights: hidden_size * 1
# - output layer biases: 1
#
# The triplet model uses the same item embedding layer
# twice and the same MLP instance twice:
# once to compute the positive similarity and the other
# time to compute the negative similarity. However because
# those two lanes in the computation graph share the same
# instances for the item embedding layer and for the MLP,
# their parameters are shared.
#
# Reminder: MLP stands for multi-layer perceptron, which is a
# common short-hand for Feed Forward Fully Connected Neural
# Network.