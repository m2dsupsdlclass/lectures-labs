def compute_loss(X,y_true):
    probas = model.forward(X)
    return model.nll(probas, y_true)

def predict_classes(X):
    probas = model.forward(X)
    return np.argmax(probas, axis=-1)

def accuracy(X, y_true):
    y_preds = predict_classes(X)
    return np.mean(y_preds == y_true)
