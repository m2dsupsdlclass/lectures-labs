from IPython.display import Image, display

predicted_batches = []
label_batches = []
n_batches = val_flow.n // batch_size
for i, (X, y) in zip(range(n_batches), val_flow):
    predicted_batches.append(model.predict(X).ravel())
    label_batches.append(y)
    print("%d/%d" % (i + 1, n_batches))

predictions = np.concatenate(predicted_batches)
true_labels = np.concatenate(label_batches)
top_offenders = np.abs(predictions - true_labels).argsort()[::-1][:10]

image_names = np.array(val_flow.filenames, dtype=np.object)[top_offenders]
for img, pred in zip(image_names, predictions[top_offenders]):
    print("predicted dog probability: %0.4f" % pred)
    display(Image(op.join(validation_folder, img)))

# Analysis:
#
# The  worst offender has the grid occlusion: this kind of grids is
# probably much more frequent for dogs in in the rest of the training
# set. This is an unwanted bias of our dataset.
#
# To fix it we would probably need to add other images with similar
# occlusion patterns to teach the model to be invariant to them.
# This could be achieved with a dedicated data augmentation scheme.
#
# The image with both a dog and a cat could clearly be considered a
# labeling error: this kind of ambiguous images should be removed
# from the validation set to properly asses the generalization ability
# of the model.
#
# The other errors are harder to understand. Introspecting the gradients
# back to the pixel space could help understand what's misleading the
# model. It could be some elements in the background that are
# statistically very correlated to dogs in the training set.