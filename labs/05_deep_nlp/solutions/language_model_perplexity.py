def perplexity(y_true, y_pred):
    """Compute the perplexity of model predictions.
    
    y_true is one-hot encoded ground truth.
    y_pred is predicted likelihoods for each class.
    
    2 ** -mean(log2(p))
    """
    likelihoods = np.sum(y_pred * y_true, axis=1)
    return 2 ** -np.mean(np.log2(likelihoods))