# %load solutions/keras_adam_and_adadelta.py
model = Sequential()
model.add(Dense(hidden_dim, input_dim=input_dim,
                activation="relu"))
model.add(Dense(hidden_dim, activation="relu"))
model.add(Dense(output_dim, activation="softmax"))

optimizer = optimizers.Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train, validation_split=0.2,
                    epochs=15, batch_size=32)
fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(12, 6))
history_df = pd.DataFrame(history.history)
history_df["epoch"] = history.epoch
history_df.plot(x="epoch", y=["loss", "val_loss"], ax=ax0)
history_df.plot(x="epoch", y=["accuracy", "val_accuracy"], ax=ax1);

# Analysis:
#
# Adam with its default global learning rate of 0.001 tends to work
# in many settings often converge as fast or faster than SGD
# with a well tuned learning rate.
# Adam adapts the learning rate locally for each neuron, this is why
# tuning its default global learning rate is rarely needed.
#
# Reference:
#
# Adam:     https://arxiv.org/abs/1412.6980
