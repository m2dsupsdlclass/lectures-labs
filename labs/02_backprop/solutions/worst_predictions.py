test_losses = -np.log(EPSILON + model.forward(X_test) * one_hot(10, y_test)).sum(axis=1)
worst_idx = test_losses.argsort()[:5]
print("test losses:", test_losses[worst_idx])
for idx in worst_idx:
    plot_prediction(model, sample_idx=idx)