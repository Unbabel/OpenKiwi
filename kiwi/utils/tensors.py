#  OpenKiwi: Open-Source Machine Translation Quality Estimation
#  Copyright (C) 2020 Unbabel <openkiwi@unbabel.com>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from kiwi.data.batch import BatchedSentence
from kiwi.data.vocabulary import Vocabulary


def pad_zeros_around_timesteps(batched_tensor: torch.Tensor) -> torch.Tensor:
    input_size = batched_tensor.size()
    left_pad = batched_tensor.new_zeros(input_size[0], 1, *input_size[2:])
    right_pad = batched_tensor.new_zeros(input_size[0], 1, *input_size[2:])
    return torch.cat((left_pad, batched_tensor, right_pad), dim=1)


def convolve_tensor(sequences, window_size, pad_value=0):
    """Convolve a sequence and apply padding.

    Arguments:
        sequences: nD tensor
        window_size: filter length
        pad_value: int value used as padding

    Return:
         (n+1)D tensor, where the last dimension has size window_size.
    """
    pad = (window_size // 2,) * 2
    pad = (0, 0) * (len(sequences.shape) - 2) + pad  # We only want to pad dim 1
    t = F.pad(sequences, pad=pad, value=pad_value)
    t = t.unfold(1, window_size, 1)

    # For 3D tensors
    # torch.nn.ConstantPad2d((0, 0, 1, 1), 0)(x).unfold(1, 3, 1)
    # F.pad(x, (0, 0, 1, 1), value=0).unfold(1, 3, 1)

    return t


# def convolve_sequence(sequence, window_size, pad_value=0):
#     """Convolve a sequence and apply padding
#
#     :param sequence: list of ids
#     :param window_size: filter length
#     :param pad_value: int value used as padding
#     :return: list of lists with size of window_size
#     """
#     pad = [pad_value for _ in range(window_size // 2)]
#     pad_sequence = pad + sequence + pad
#     return list(windowed(pad_sequence, window_size, fillvalue=pad_value))


def apply_packed_sequence(rnn, padded_sequences, lengths):
    """Run a forward pass of padded_sequences through an rnn using packed sequence.

    Arguments:
       rnn: The RNN that that we want to compute a forward pass with.
       padded_sequences (FloatTensor b x seq x dim): A batch of padded_sequences.
       lengths (LongTensor batch): The length of each sequence in the batch.

    Return:
       output: the output of the RNN `rnn` with input `padded_sequences`
    """
    # Sort Batch by sequence length
    total_length = padded_sequences.size(1)  # Get the max sequence length
    lengths_sorted, permutation = torch.sort(lengths, descending=True)
    padded_sequences_sorted = padded_sequences[permutation]

    # Use Packed Sequence
    padded_sequences_packed = pack(
        padded_sequences_sorted, lengths_sorted, batch_first=True
    )
    outputs_packed, (hidden, cell) = rnn(padded_sequences_packed)
    outputs_sorted, _ = unpack(
        outputs_packed, batch_first=True, total_length=total_length
    )

    # Restore original order
    _, permutation_rev = torch.sort(permutation, descending=False)
    outputs = outputs_sorted[permutation_rev]
    hidden, cell = hidden[:, permutation_rev], cell[:, permutation_rev]
    return outputs, (hidden, cell)


def replace_token(target: torch.LongTensor, old: int, new: int):
    """Replace old tokens with new.

    Arguments:
        target
        old: the token to be replaced by new.
        new: the token used to replace old.

    """
    return target.masked_fill(target == old, new)


def make_classes_loss_weights(vocab: Vocabulary, label_weights: Dict[str, float]):
    """Create a loss weight vector for nn.CrossEntropyLoss.

    Arguments:
        vocab: vocabulary for classes.
        label_weights: weight for specific classes (str); classes in vocab and not in
                       this dict will get a weight of 1.

    Return:
       weights (FloatTensor): weight Tensor of shape `nb_classes`.
    """
    nb_classes = (vocab.net_length(),)
    class_weights = torch.ones(nb_classes)
    for class_label, weight in label_weights.items():
        class_idx = vocab.stoi[class_label]
        class_weights[class_idx] = weight
    return class_weights


def sequence_mask(lengths: torch.LongTensor, max_len: Optional[int] = None):
    """Create a boolean mask from sequence lengths.

    Arguments:
        lengths: lengths with shape (bs,)
        max_len: max sequence length; if None it will be set to lengths.max()
    """
    if max_len is None:
        max_len = lengths.max()
    # aranges = torch.arange(max_len).repeat(lengths.size(0), 1)
    # mask = aranges < lengths.unsqueeze(1)
    # This is equivalent
    mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
    return mask


def unmask(tensor, mask):
    """Unmask a tensor and convert it back to a list of lists."""
    lengths = mask.int().sum(dim=-1).tolist()
    return [x[: lengths[i]].tolist() for i, x in enumerate(tensor)]


def unsqueeze_as(tensor, as_tensor, dim=-1):
    """Expand new dimensions based on a template tensor along `dim` axis."""
    x = tensor
    while x.dim() < as_tensor.dim():
        x = x.unsqueeze(dim)
    return x


def make_mergeable_tensors(t1: torch.Tensor, t2: torch.Tensor):
    """Expand a new dimension in t1 and t2 and expand them so that both
    tensors will have the same number of timesteps.

    Arguments:
        t1: tensor with shape (bs, ..., m, d1)
        t2: tensor with shape (bs, ..., n, d2)

    Return:
        tuple of
            torch.Tensor: (bs, ..., m, n, d1),
            torch.Tensor: (bs, ..., m, n, d2)
    """
    assert t1.dim() == t2.dim()
    assert t1.dim() >= 3
    assert t1.shape[:-2] == t2.shape[:-2]
    # new_shape = [-1, ..., m, n, -1]
    new_shape = [-1 for _ in range(t1.dim() + 1)]
    new_shape[-3] = t1.shape[-2]  # m
    new_shape[-2] = t2.shape[-2]  # n
    # (bs, ..., m, d1) -> (bs, ..., m, 1, d1) -> (bs, ..., m, n, d1)
    new_t1 = t1.unsqueeze(-2).expand(new_shape)
    # (bs, ..., n, d2) -> (bs, ..., 1, n, d2) -> (bs, ..., m, n, d2)
    new_t2 = t2.unsqueeze(-3).expand(new_shape)
    return new_t1, new_t2


class GradientMul(Function):
    @staticmethod
    def forward(ctx, x, constant=0):
        ctx.constant = constant
        return x

    @staticmethod
    def backward(ctx, grad):
        return ctx.constant * grad, None


gradient_mul = GradientMul.apply


def feedforward(
    in_dim,
    n_layers,
    shrink=2,
    out_dim=None,
    activation=nn.Tanh,
    final_activation=False,
    dropout=0.0,
):
    """Constructor for FeedForward Layers"""
    dim = in_dim
    module_dict = OrderedDict()
    for layer_i in range(n_layers - 1):
        next_dim = dim // shrink
        module_dict['linear_{}'.format(layer_i)] = nn.Linear(dim, next_dim)
        module_dict['activation_{}'.format(layer_i)] = activation()
        module_dict['dropout_{}'.format(layer_i)] = nn.Dropout(dropout)
        dim = next_dim
    next_dim = out_dim or (dim // 2)
    module_dict['linear_{}'.format(n_layers - 1)] = nn.Linear(dim, next_dim)
    if final_activation:
        module_dict['activation_{}'.format(n_layers - 1)] = activation()
    return nn.Sequential(module_dict)


def retrieve_tokens_mask(input_batch: BatchedSentence):
    """Compute Mask of Tokens for side.

    Migrated from FieldEmbedder.get_mask()

    Arguments:
        input_batch (BatchedSentence): batch of tensors

    Return:
        mask tensor
    """
    assert isinstance(input_batch, BatchedSentence)

    tensor = input_batch.tensor
    lengths = input_batch.lengths
    mask = torch.ones_like(tensor, dtype=torch.int)
    mask[:] = torch.arange(mask.shape[1])
    mask = mask < lengths.unsqueeze(-1).int()
    return mask


def select_positions(tensor, indices):
    range_vector = torch.arange(tensor.size(0), device=tensor.device).unsqueeze(1)
    return tensor[range_vector, indices]


def pieces_to_tokens(features_tensor, batch, strategy='first'):
    """Join together pieces of a token back into the original token dimension."""
    if strategy == 'first':
        # Use the bounds for the wordpieces
        # This discards all features not from the first wordpiece of the token
        return select_positions(features_tensor, batch.bounds)
    else:
        raise NotImplementedError('Only available joining strategy is first.')
