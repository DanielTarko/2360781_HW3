import re
import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from typing import Iterator


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    #  Create two maps as described in the docstring above.
    #  It's best if you also sort the chars before assigning indices, so that
    #  they're in lexical order.
    # ====== YOUR CODE: ======
    unique_chars = sorted(set(text))
    char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
    idx_to_char = {idx: char for idx, char in enumerate(unique_chars)}    
    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # ====== YOUR CODE: ======
    chars_to_remove_set = set(chars_to_remove)
    text_clean = ''.join(char for char in text if char not in chars_to_remove_set)
    n_removed = len(text) - len(text_clean)
    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tensor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # ====== YOUR CODE: ======
    num_chars = len(char_to_idx)
    onehot_tensor = torch.zeros((len(text), num_chars), dtype=torch.int8)
    
    for i, char in enumerate(text):
        if char in char_to_idx:
            onehot_tensor[i, char_to_idx[char]] = 1
    result = onehot_tensor
    # ========================
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # ====== YOUR CODE: ======
    chars = ''
    for row in embedded_text:
        idx = torch.argmax(row).item()
        chars += idx_to_char[idx]
    # ========================
    return chars


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int, device="cpu"):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    #  Implement the labelled samples creation.
    #  1. Embed the given text.
    #  2. Create the samples tensor by splitting to groups of seq_len.
    #     Notice that the last char has no label, so don't use it.
    #  3. Create the labels tensor in a similar way and convert to indices.
    #  Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======
    num_chars = len(char_to_idx)
    num_samples = (len(text) - 1) // seq_len

    samples = torch.zeros((num_samples, seq_len, num_chars), dtype=torch.int8, device=device)
    labels = torch.zeros((num_samples, seq_len), dtype=torch.long, device=device)

    for i in range(num_samples):
        sample_text = text[i * seq_len:(i + 1) * seq_len]
        label_text = text[i * seq_len + 1:(i + 1) * seq_len + 1]

        for j, char in enumerate(sample_text):
            samples[i, j, char_to_idx[char]] = 1

        for j, char in enumerate(label_text):
            labels[i, j] = char_to_idx[char]
    # ========================
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======
    scaled = y / temperature
    
    # Subtract the maximum value for numerical stability
    # This prevents overflow when computing exp
    max_val = scaled.max(dim=dim, keepdim=True)[0]
    exp_scaled = (scaled - max_val).exp()
    
    # Compute sum along the specified dimension for normalization
    sum_exp = exp_scaled.sum(dim=dim, keepdim=True)
    
    # Compute softmax: exp(x_i/T) / sum(exp(x_j/T))
    result = exp_scaled / sum_exp    
    # ========================
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO:
    #  Implement char-by-char text generation.
    #  1. Feed the start_sequence into the model.
    #  2. Sample a new char from the output distribution of the last output
    #     char. Convert output to probabilities first.
    #     See torch.multinomial() for the sampling part.
    #  3. Feed the new char into the model.
    #  4. Rinse and Repeat.
    #  Note that tracking tensor operations for gradient calculation is not
    #  necessary for this. Best to disable tracking for speed.
    #  See torch.no_grad().
    # ====== YOUR CODE: ======
    # Convert start sequence to tensor of character indices
    char_sequence = torch.tensor(
        [char_to_idx[ch] for ch in start_sequence],
        dtype=torch.long,
        device=device
    )
    
    # Create one-hot encoded input sequence
    vocab_size = len(char_to_idx)
    x = torch.zeros((1, len(start_sequence), vocab_size), device=device)
    x[0, torch.arange(len(start_sequence)), char_sequence] = 1
    
    with torch.no_grad():  # Disable gradient tracking for generation
        # Get initial prediction and hidden state from start sequence
        hidden_state = None
        y, hidden_state = model(x, hidden_state)
        
        # Generate remaining characters one by one
        while len(out_text) < n_chars:
            # Get probability distribution for next character
            logits = y[0, -1, :] / T  # Apply temperature scaling
            probs = torch.softmax(logits, dim=0)
            
            # Sample next character from the distribution
            next_char_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = idx_to_char[next_char_idx]
            out_text += next_char
            
            # Prepare input for next iteration
            x = torch.zeros((1, 1, vocab_size), device=device)
            x[0, 0, next_char_idx] = 1
            
            # Get prediction for next character
            y, hidden_state = model(x, hidden_state)
    # ========================

    return out_text


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        #  Return an iterator of indices, i.e. numbers in range(len(dataset)).
        #  dataset and represents one  batch.
        #  The indices must be generated in a way that ensures
        #  that when a batch of size self.batch_size of indices is taken, samples in
        #  the same index of adjacent batches are also adjacent in the dataset.
        #  In the case when the last batch can't have batch_size samples,
        #  you can drop it.
          # idx should be a 1-d list of indices.
        # ====== YOUR CODE: ======
        import math
        num_batches = math.floor(len(self.dataset)/self.batch_size)

        idx = [0] * (num_batches*self.batch_size)
        for i in range(len(self.dataset)):
            if i < num_batches*self.batch_size:
                idx[(self.batch_size*(i%num_batches)) + math.floor(i/num_batches)] = i
        # ========================
        return iter(idx)

    def __len__(self):
        return len(self.dataset)


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []

        # TODO: READ THIS SECTION!!

        # ====== YOUR CODE: ======
        # Create parameter matrices for each layer
        for layer in range(n_layers):
            layer_in_dim = in_dim if layer == 0 else h_dim
            
            # Update gate parameters
            self.add_module(f'Wxz_{layer}', nn.Linear(layer_in_dim, h_dim, bias=True))
            self.add_module(f'Whz_{layer}', nn.Linear(h_dim, h_dim, bias=False))
            
            # Reset gate parameters
            self.add_module(f'Wxr_{layer}', nn.Linear(layer_in_dim, h_dim, bias=True))
            self.add_module(f'Whr_{layer}', nn.Linear(h_dim, h_dim, bias=False))
            
            # Candidate state parameters
            self.add_module(f'Wxg_{layer}', nn.Linear(layer_in_dim, h_dim, bias=True))
            self.add_module(f'Whg_{layer}', nn.Linear(h_dim, h_dim, bias=False))
        
        # Output layer
        self.Why = nn.Linear(h_dim, out_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(
                    torch.zeros(batch_size, self.h_dim, device=input.device)
                )
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None

        # TODO: READ THIS SECTION!!
        # ====== YOUR CODE: ======
        # Loop over layers of the model
        # Process each timestep
        for t in range(seq_len):
            # Current input for all batches at this timestep
            xt = layer_input[:, t, :]
            
            # Process through each layer
            current_input = xt
            for layer in range(self.n_layers):
                # Get layer parameters
                Wxz = getattr(self, f'Wxz_{layer}')
                Whz = getattr(self, f'Whz_{layer}')
                Wxr = getattr(self, f'Wxr_{layer}')
                Whr = getattr(self, f'Whr_{layer}')
                Wxg = getattr(self, f'Wxg_{layer}')
                Whg = getattr(self, f'Whg_{layer}')
                
                h_prev = layer_states[layer]
                
                # Update gate
                zt = torch.sigmoid(
                    Wxz(current_input) + Whz(h_prev)
                )
                
                # Reset gate
                rt = torch.sigmoid(
                    Wxr(current_input) + Whr(h_prev)
                )
                
                # Candidate state
                gt = torch.tanh(
                    Wxg(current_input) + Whg(rt * h_prev)
                )
                
                # New hidden state
                ht = zt * h_prev + (1 - zt) * gt
                
                # Update layer state
                layer_states[layer] = ht
                
                # Prepare input for next layer
                current_input = ht
                if self.dropout is not None and layer < self.n_layers - 1:
                    current_input = self.dropout(current_input)
            
            # Store output for this timestep
            if layer_output is None:
                layer_output = torch.zeros(
                    batch_size, seq_len, self.out_dim, 
                    device=input.device
                )
            layer_output[:, t, :] = self.Why(current_input)

        # Prepare final hidden state tensor
        hidden_state = torch.stack([state for state in layer_states], dim=1)
        # ========================
        return layer_output, hidden_state
