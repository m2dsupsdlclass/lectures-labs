n_sequences = len(sequences)
n_sequences_test = len(sequences_test)
voc_size = len(chars)

X = np.zeros((n_sequences, max_length, voc_size),
             dtype=np.float32)
y = np.zeros((n_sequences, voc_size), dtype=np.float32)

X_test = np.zeros((n_sequences_test, max_length, voc_size),
                  dtype=np.float32)
y_test = np.zeros((n_sequences_test, voc_size), dtype=np.float32)


for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
    
for i, sequence in enumerate(sequences_test):
    for t, char in enumerate(sequence):
        X_test[i, t, char_indices[char]] = 1
    y_test[i, char_indices[next_chars_test[i]]] = 1