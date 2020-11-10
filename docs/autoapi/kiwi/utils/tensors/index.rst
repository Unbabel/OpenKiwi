:mod:`kiwi.utils.tensors`
=========================

.. py:module:: kiwi.utils.tensors


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.utils.tensors.GradientMul



Functions
~~~~~~~~~

.. autoapisummary::

   kiwi.utils.tensors.pad_zeros_around_timesteps
   kiwi.utils.tensors.convolve_tensor
   kiwi.utils.tensors.apply_packed_sequence
   kiwi.utils.tensors.replace_token
   kiwi.utils.tensors.make_classes_loss_weights
   kiwi.utils.tensors.sequence_mask
   kiwi.utils.tensors.unmask
   kiwi.utils.tensors.unsqueeze_as
   kiwi.utils.tensors.make_mergeable_tensors
   kiwi.utils.tensors.retrieve_tokens_mask
   kiwi.utils.tensors.select_positions
   kiwi.utils.tensors.pieces_to_tokens


.. function:: pad_zeros_around_timesteps(batched_tensor: torch.Tensor) -> torch.Tensor


.. function:: convolve_tensor(sequences, window_size, pad_value=0)

   Convolve a sequence and apply padding.

   :param sequences: nD tensor
   :param window_size: filter length
   :param pad_value: int value used as padding

   :returns: (n+1)D tensor, where the last dimension has size window_size.


.. function:: apply_packed_sequence(rnn, padded_sequences, lengths)

   Run a forward pass of padded_sequences through an rnn using packed sequence.

   :param rnn: The RNN that that we want to compute a forward pass with.
   :param padded_sequences: A batch of padded_sequences.
   :type padded_sequences: FloatTensor b x seq x dim
   :param lengths: The length of each sequence in the batch.
   :type lengths: LongTensor batch

   :returns: the output of the RNN `rnn` with input `padded_sequences`
   :rtype: output


.. function:: replace_token(target: torch.LongTensor, old: int, new: int)

   Replace old tokens with new.

   :param target:
   :param old: the token to be replaced by new.
   :param new: the token used to replace old.


.. function:: make_classes_loss_weights(vocab: Vocabulary, label_weights: Dict[str, float])

   Create a loss weight vector for nn.CrossEntropyLoss.

   :param vocab: vocabulary for classes.
   :param label_weights: weight for specific classes (str); classes in vocab and not in
                         this dict will get a weight of 1.

   :returns: weight Tensor of shape `nb_classes`.
   :rtype: weights (FloatTensor)


.. function:: sequence_mask(lengths: torch.LongTensor, max_len: Optional[int] = None)

   Create a boolean mask from sequence lengths.

   :param lengths: lengths with shape (bs,)
   :param max_len: max sequence length; if None it will be set to lengths.max()


.. function:: unmask(tensor, mask)

   Unmask a tensor and convert it back to a list of lists.


.. function:: unsqueeze_as(tensor, as_tensor, dim=-1)

   Expand new dimensions based on a template tensor along `dim` axis.


.. function:: make_mergeable_tensors(t1: torch.Tensor, t2: torch.Tensor)

   Expand a new dimension in t1 and t2 and expand them so that both
   tensors will have the same number of timesteps.

   :param t1: tensor with shape (bs, ..., m, d1)
   :param t2: tensor with shape (bs, ..., n, d2)

   :returns:

             tuple of
                 torch.Tensor: (bs, ..., m, n, d1),
                 torch.Tensor: (bs, ..., m, n, d2)


.. py:class:: GradientMul

   Bases: :class:`torch.autograd.Function`

   Records operation history and defines formulas for differentiating ops.

   See the Note on extending the autograd engine for more details on how to use
   this class: https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd

   Every operation performed on :class:`Tensor` s creates a new function
   object, that performs the computation, and records that it happened.
   The history is retained in the form of a DAG of functions, with edges
   denoting data dependencies (``input <- output``). Then, when backward is
   called, the graph is processed in the topological ordering, by calling
   :func:`backward` methods of each :class:`Function` object, and passing
   returned gradients on to next :class:`Function` s.

   Normally, the only way users interact with functions is by creating
   subclasses and defining new operations. This is a recommended way of
   extending torch.autograd.

   Examples::

       >>> class Exp(Function):
       >>>
       >>>     @staticmethod
       >>>     def forward(ctx, i):
       >>>         result = i.exp()
       >>>         ctx.save_for_backward(result)
       >>>         return result
       >>>
       >>>     @staticmethod
       >>>     def backward(ctx, grad_output):
       >>>         result, = ctx.saved_tensors
       >>>         return grad_output * result
       >>>
       >>> #Use it by calling the apply method:
       >>> output = Exp.apply(input)

   .. method:: forward(ctx, x, constant=0)
      :staticmethod:

      Performs the operation.

      This function is to be overridden by all subclasses.

      It must accept a context ctx as the first argument, followed by any
      number of arguments (tensors or other types).

      The context can be used to store tensors that can be then retrieved
      during the backward pass.


   .. method:: backward(ctx, grad)
      :staticmethod:

      Defines a formula for differentiating the operation.

      This function is to be overridden by all subclasses.

      It must accept a context :attr:`ctx` as the first argument, followed by
      as many outputs did :func:`forward` return, and it should return as many
      tensors, as there were inputs to :func:`forward`. Each argument is the
      gradient w.r.t the given output, and each returned value should be the
      gradient w.r.t. the corresponding input.

      The context can be used to retrieve tensors saved during the forward
      pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
      of booleans representing whether each input needs gradient. E.g.,
      :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
      first input to :func:`forward` needs gradient computated w.r.t. the
      output.



.. data:: gradient_mul
   

   

.. function:: retrieve_tokens_mask(input_batch: BatchedSentence)

   Compute Mask of Tokens for side.

   Migrated from FieldEmbedder.get_mask()

   :param input_batch: batch of tensors
   :type input_batch: BatchedSentence

   :returns: mask tensor


.. function:: select_positions(tensor, indices)


.. function:: pieces_to_tokens(features_tensor, batch, strategy='first')

   Join together pieces of a token back into the original token dimension.


