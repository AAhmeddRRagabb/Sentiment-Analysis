###############################################################################
# This file contains the required functions to preprocess the data
# الله المستعان
###############################################################################

import re

def tokenize_text(text):
    # separate words & puncs
    text = re.sub(r"([.?!,])", r" \1 ", text) 
    # Tokenize
    tokens = [word for word in text.split()]
    return tokens


def pad_sequence(sequence: list, max_len, padding_value = "<pad>"):
    """
    This function pads a given sequence of strings to a max_len length

    Args:
        sequence     : a list contains the sequence
        max_len      : The value to pad the sequence to
        padding_value: The value to pad with
    """
    if len(sequence) > max_len:
        return sequence[:max_len]
        
    elif len(sequence) < max_len:
        num_of_paddings = max_len - len(sequence)
        return sequence + [padding_value] * num_of_paddings
    else:
        return sequence