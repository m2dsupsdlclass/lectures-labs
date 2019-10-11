prediction = simple_seq2seq(first_test_sequence).numpy()
print("prediction shape:", prediction.shape)

# Let's use `argmax` to extract the predicted token ids at each step:
predicted_token_ids = prediction[0].argmax(-1)
print("prediction token ids:", predicted_token_ids)

# We can use the shared reverse vocabulary to map
# this back to the string representation of the tokens,
# as well as removing Padding and EOS symbols
predicted_numbers = [rev_shared_vocab[token_id] for token_id in predicted_token_ids
                     if token_id not in (shared_vocab[PAD], shared_vocab[EOS])]
print("predicted number:", "".join(predicted_numbers))
print("test number:", num_test[0])

# The model successfully predicted the test sequence.
# However, we provided the full sequence as input, including all the solution
# (except for the last number). In a real testing condition, one wouldn't
# have the full input sequence, but only what is provided before the "GO"
# symbol
