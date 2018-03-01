def make_input_output(source_tokens, target_tokens, reverse_source=True):
    if reverse_source:
        source_tokens = list(reversed(source_tokens))
    input_tokens = source_tokens + [GO] + target_tokens
    output_tokens = target_tokens + [EOS]
    return input_tokens, output_tokens