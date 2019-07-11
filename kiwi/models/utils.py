#  OpenKiwi: Open-Source Machine Translation Quality Estimation
#  Copyright (C) 2019 Unbabel <openkiwi@unbabel.com>
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

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from more_itertools import first, flatten
from torch.autograd import Function
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from kiwi import constants as const

logger = logging.getLogger(__name__)


def unroll(list_of_lists):
    """
    :param list_of_lists: a list that contains lists
    :param rec: unroll recursively
    :return: a flattened list
    """
    if isinstance(first(list_of_lists), (np.ndarray, list)):
        return list(flatten(list_of_lists))
    return list_of_lists


def convolve_tensor(sequences, window_size, pad_value=0):
    """Convolve a sequence and apply padding

    :param sequence: 2D tensor
    :param window_size: filter length
    :param pad_value: int value used as padding
    :return: 3D tensor, where the last dimension has size window_size
    """
    pad = (window_size // 2,) * 2
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


def align_tensor(
    tensor,
    alignments,
    max_aligned,
    unaligned_idx,
    padding_idx,
    pad_size,
    target_length=None,
):
    alignments = [
        map_alignments_to_target(sample, target_length=target_length)
        for sample in alignments
    ]

    # aligned_tensor = tensor.new_full(
    #     (tensor.shape[0], pad_size, max_aligned, tensor.shape[2]),
    #     padding_idx)

    aligned = [
        align_source(
            sample, alignment, max_aligned, unaligned_idx, padding_idx, pad_size
        )
        for sample, alignment in zip(tensor, alignments)
    ]
    aligned_tensor = torch.stack([sample[0] for sample in aligned])
    nb_alignments = torch.stack([sample[1] for sample in aligned])
    return aligned_tensor, nb_alignments


def map_alignments_to_target(src2tgt_alignments, target_length=None):
    """Maps a target index to a list of source indexes.

    Args:
        src2tgt_alignments (list): list of tuples with source, target indexes.
        target_length: size of the target side; if None, the highest index
            in the alignments is used.

    Returns:
        A list of size target_length where position i refers to the i-th
        target token and contains a list of source indexes aligned to it.

    """
    if target_length is None:
        if not src2tgt_alignments:
            target_length = 0
        else:
            target_length = 1 + max(src2tgt_alignments, key=lambda a: a[1])[1]

    trg2src = [None] * target_length
    for source, target in src2tgt_alignments:
        if not trg2src[target]:
            trg2src[target] = []
        trg2src[target].append(source)
    return trg2src


def align_source(
    source,
    trg2src_alignments,
    max_aligned,
    unaligned_idx,
    padding_idx,
    pad_size,
):
    assert len(source.shape) == 2
    window_size = source.shape[1]

    assert len(trg2src_alignments) <= pad_size
    aligned_source = source.new_full(
        (pad_size, max_aligned, window_size), padding_idx
    )
    unaligned = source.new_full((window_size,), unaligned_idx)
    nb_alignments = source.new_ones(pad_size, dtype=torch.float)

    for i, source_positions in enumerate(trg2src_alignments):
        if not source_positions:
            aligned_source[i, 0] = unaligned
        else:
            selected = torch.index_select(
                source,
                0,
                torch.tensor(
                    source_positions[:max_aligned], device=source.device
                ),
            )
            aligned_source[i, : len(selected)] = selected
            # counts how many tokens is a target token aligned to
            nb_alignments[i] = len(selected)
    return aligned_source, nb_alignments


def apply_packed_sequence(rnn, embedding, lengths):
    """ Runs a forward pass of embeddings through an rnn using packed sequence.
    Args:
       rnn: The RNN that that we want to compute a forward pass with.
       embedding (FloatTensor b x seq x dim): A batch of sequence embeddings.
       lengths (LongTensor batch): The length of each sequence in the batch.

    Returns:
       output: The output of the RNN `rnn` with input `embedding`
    """
    # Sort Batch by sequence length
    lengths_sorted, permutation = torch.sort(lengths, descending=True)
    embedding_sorted = embedding[permutation]

    # Use Packed Sequence
    embedding_packed = pack(embedding_sorted, lengths_sorted, batch_first=True)
    outputs_packed, (hidden, cell) = rnn(embedding_packed)
    outputs_sorted, _ = unpack(outputs_packed, batch_first=True)
    # Restore original order
    _, permutation_rev = torch.sort(permutation, descending=False)
    outputs = outputs_sorted[permutation_rev]
    hidden, cell = hidden[:, permutation_rev], cell[:, permutation_rev]
    return outputs, (hidden, cell)


def replace_token(target, old, new):
    """Replaces old tokens with new.

    args: target (LongTensor)
          old (int): The token to be replaced by new
          new (int): The token used to replace old

    """
    return target.masked_fill(target == old, new)


def make_loss_weights(nb_classes, target_idx, weight):
    """Creates a loss weight vector for nn.CrossEntropyLoss

    args:
        nb_classes: Number of classes
        target_idx: ID of the target (reweighted) class
        weight: Weight of the target class

    returns:
       weights (FloatTensor): Weight Tensor of shape `nb_classes` such that
                                  `weights[target_idx] = weight`
                                  `weights[other_idx] = 1.0`
    """

    weights = torch.ones(nb_classes)
    weights[target_idx] = weight
    return weights


def load_torch_file(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError('Torch file not found: {}'.format(file_path))

    file_dict = torch.load(
        str(file_path), map_location=lambda storage, loc: storage
    )
    if isinstance(file_dict, Path):
        # Resolve cases where file is just a link to another torch file
        linked_path = file_dict
        if not linked_path.exists():
            relative_path = (
                file_path.with_name(file_dict.name) / const.MODEL_FILE
            )
            if relative_path.exists():
                linked_path = relative_path
        return load_torch_file(linked_path)
    return file_dict


class GradientMul(Function):
    @staticmethod
    def forward(ctx, x, constant=0):
        ctx.constant = constant
        return x

    @staticmethod
    def backward(ctx, grad):
        return ctx.constant * grad, None


gradient_mul = GradientMul.apply
