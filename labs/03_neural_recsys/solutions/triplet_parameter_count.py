# Analysis:
#
# Both models have exactly the same number of parameters,
# namely the parameters of the 2 embeddings:
#
# - user embedding: n_users x embedding_dim
# - item embedding: n_items x embedding_dim
#
# The triplet model uses the same item embedding twice,
# once to compute the positive similarity and the other
# time to compute the negative similarity. However because
# those two nodes in the computation graph share the same
# instance of the item embedding layer, the item embedding
# weight matrix is shared by the two branches of the
# graph and therefore the total number of parameters for
# each model is in both cases:
#
# (n_users x embedding_dim) + (n_items x embedding_dim)