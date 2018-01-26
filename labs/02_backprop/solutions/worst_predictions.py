test_losses = -np.sum(np.log(EPSILON + model.forward(X_test))
                      * one_hot(10, y_test), axis=1)

# Sort by ascending loss: best predictions first, worst
# at the end
ranked_by_loss = test_losses.argsort()

# Extract and display the top 5 worst predictions at
# the end:
worst_idx = ranked_by_loss[-5:]
print("test losses:", test_losses[worst_idx])
for idx in worst_idx:
    plot_prediction(model, sample_idx=idx)