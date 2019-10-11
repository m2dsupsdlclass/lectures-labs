from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, GRU, Dense

vocab_size = len(shared_vocab)
simple_seq2seq = Sequential()
simple_seq2seq.add(Embedding(vocab_size, 32, input_length=max_length))
simple_seq2seq.add(Dropout(0.2))
simple_seq2seq.add(GRU(256, return_sequences=True, reset_after=False))
# Reset_after = True needed for compatibility with the GPU pretrained model
# if not working on CPU, you may switch it to True

simple_seq2seq.add(Dense(vocab_size, activation='softmax'))

# Here we use the sparse_categorical_crossentropy loss to be able to pass
# integer-coded output for the token ids without having to convert to one-hot
# codes
simple_seq2seq.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
