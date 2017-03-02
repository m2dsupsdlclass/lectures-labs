X = np.zeros((len(sequences), max_length, len(chars)), dtype=np.float32)
y = np.zeros((len(sequences), len(chars)), dtype=np.float32)

for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1