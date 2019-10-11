def beam_translate(model, source_sequence, shared_vocab, rev_shared_vocab,
                   word_level_source=True, word_level_target=True,
                   beam_size=10, return_ll=False):
    """Decode candidate translations with a beam search strategy

    If return_ll is False, only the best candidate string is returned.
    If return_ll is True, all the candidate strings and their loglikelihoods
    are returned.
    """
    # Initialize the list of input token ids with the source sequence
    source_tokens = tokenize(source_sequence, word_level=word_level_source)
    input_ids = [shared_vocab.get(t, UNK) for t in source_tokens[::-1]]
    input_ids += [shared_vocab[GO]]

    # initialize loglikelihood, input token ids, decoded tokens for
    # each candidate in the beam
    candidates = [(0, input_ids[:], [], False)]

    # Prepare a fixed size numpy array that matches the expected input
    # shape for the model
    input_array = np.empty(shape=(beam_size, model.input_shape[1]),
                           dtype=np.int32)
    while any([not done and (len(input_ids) < max_length)
               for _, input_ids, _, done in candidates]):
        # Vectorize a the list of input tokens and use zeros padding.
        input_array.fill(shared_vocab[PAD])
        for i, (_, input_ids, _, done) in enumerate(candidates):
            if not done:
                input_array[i, -len(input_ids):] = input_ids

        # Predict the next output in a single call to the model to amortize
        # the overhead and benefit from vector data parallelism on GPU.
        next_likelihood_batch = model(input_array).numpy()

        # Build the new candidates list by summing the loglikelood of the
        # next token with their parents for each new possible expansion.
        new_candidates = []
        for i, (ll, input_ids, decoded, done) in enumerate(candidates):
            if done:
                new_candidates.append((ll, input_ids, decoded, done))
            else:
                next_loglikelihoods = np.log(next_likelihood_batch[i, -1])
                for next_token_id, next_ll in enumerate(next_loglikelihoods):
                    new_ll = ll + next_ll
                    new_input_ids = input_ids[:]
                    new_input_ids.append(next_token_id)
                    new_decoded = decoded[:]
                    new_done = done
                    if next_token_id == shared_vocab[EOS]:
                        new_done = True
                    if not new_done:
                        new_decoded.append(rev_shared_vocab[next_token_id])
                    new_candidates.append(
                        (new_ll, new_input_ids, new_decoded, new_done))

        # Only keep a beam of the most promising candidates
        new_candidates.sort(reverse=True)
        candidates = new_candidates[:beam_size]

    separator = " " if word_level_target else ""
    if return_ll:
        return [(separator.join(decoded), ll) for ll, _, decoded, _ in candidates]
    else:
        _, _, decoded, done = candidates[0]
        return separator.join(decoded)
