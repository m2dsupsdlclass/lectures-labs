from tensorflow.keras.models import Sequential

model3 = Sequential([
    embedding_layer,
    Flatten(),
])

labels_to_encode = np.array([[3]])
print(model3.predict(labels_to_encode))
