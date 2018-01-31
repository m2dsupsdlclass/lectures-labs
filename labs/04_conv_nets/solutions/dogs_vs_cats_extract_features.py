from time import time

features = []
labels = []

t0 = time()
count = 0
for X, y in train_flow:
    labels.append(y)
    features.append(base_model.predict(X))
    count += len(y)
    if count % 100 == 0:
        print("processed %d images at %d images/s"
              % (count, count / (time() - t0)))
    if count >= 5000:
        break

labels_train = np.concatenate(labels)
features_train = np.vstack(features)
np.save('labels_train.npy', labels_train)
np.save('features_train.npy', features_train)