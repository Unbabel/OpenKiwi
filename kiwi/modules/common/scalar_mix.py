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
import torch
from torch.nn import Parameter, ParameterList


class ScalarMixWithDropout(torch.nn.Module):
    """Compute a parameterised scalar mixture of N tensors.

    :math:`mixture = \\gamma * \\sum(s_k * tensor_k)`,
    where :math:`s = softmax(w)`, with :math:`w` and :math:`gamma` scalar parameters.

    If ``do_layer_norm=True``, then apply layer normalization to each tensor before
    weighting.

    If ``dropout > 0``, then for each scalar weight, adjust its softmax weight mass to 0
    with the dropout probability (i.e., setting the unnormalized weight to -inf).
    This effectively should redistribute dropped probability mass to all other weights.

    Original implementation:
        - https://github.com/Hyperparticle/udify
    Copied from COMET:
        - https://gitlab.com/Unbabel/language-technologies/unbabel-comet
    """

    def __init__(
        self,
        mixture_size: int,
        do_layer_norm: bool = False,
        initial_scalar_parameters: list = None,
        trainable: bool = True,
        dropout: float = None,
        dropout_value: float = -1e20,
    ) -> None:
        super(ScalarMixWithDropout, self).__init__()
        self.mixture_size = mixture_size
        self.do_layer_norm = do_layer_norm
        self.dropout = dropout

        if initial_scalar_parameters is None:
            initial_scalar_parameters = [0.0] * mixture_size
        elif len(initial_scalar_parameters) != mixture_size:
            raise Exception(
                "Length of initial_scalar_parameters {} differs \
                from mixture_size {}".format(
                    initial_scalar_parameters, mixture_size
                )
            )

        self.scalar_parameters = ParameterList(
            [
                Parameter(
                    torch.FloatTensor([initial_scalar_parameters[i]]),
                    requires_grad=trainable,
                )
                for i in range(mixture_size)
            ]
        )

        self.gamma = Parameter(torch.FloatTensor([1.0]), requires_grad=trainable)

        if self.dropout:
            dropout_mask = torch.zeros(len(self.scalar_parameters))
            dropout_fill = torch.empty(len(self.scalar_parameters)).fill_(dropout_value)
            self.register_buffer("dropout_mask", dropout_mask)
            self.register_buffer("dropout_fill", dropout_fill)

    def forward(
        self,
        tensors: list,  # pylint: disable=arguments-differ
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute a weighted average of the 'tensors'.

        The input tensors can be any shape with at least two dimensions, but must all
        have the same shape.

        When ``do_layer_norm=True``, ``mask`` is required. If ``tensors`` have
        dimensions ``(dim_0, ..., dim_{n-1}, dim_n)``, then ``mask`` should have dims
        ``(dim_0, ..., dim_{n-1})``, as in the typical case with ``tensors`` of shape
        ``(batch_size, timesteps, dim)`` and ``mask`` of shape
        ``(batch_size, timesteps)``.
        """
        if len(tensors) != self.mixture_size:
            raise Exception(
                f"{len(tensors)} tensors were passed, but the module was initialized "
                f"to mix {self.mixture_size} tensors."
            )

        def _do_layer_norm(tensor, broadcast_mask, num_elements_not_masked):
            tensor_masked = tensor * broadcast_mask
            mean = torch.sum(tensor_masked) / num_elements_not_masked
            variance = (
                torch.sum(((tensor_masked - mean) * broadcast_mask) ** 2)
                / num_elements_not_masked
            )
            return (tensor - mean) / torch.sqrt(variance + 1e-12)

        weights = torch.cat([parameter for parameter in self.scalar_parameters])

        if self.training and self.dropout:
            weights = torch.where(
                self.dropout_mask.uniform_() > self.dropout, weights, self.dropout_fill
            )

        normed_weights = torch.nn.functional.softmax(weights, dim=0)
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)

        if not self.do_layer_norm:
            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * tensor)
            return self.gamma * sum(pieces)

        else:
            mask_float = mask.float()
            broadcast_mask = mask_float.unsqueeze(-1)
            input_dim = tensors[0].size(-1)
            num_elements_not_masked = torch.sum(mask_float) * input_dim

            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(
                    weight
                    * _do_layer_norm(tensor, broadcast_mask, num_elements_not_masked)
                )
            return self.gamma * sum(pieces)
