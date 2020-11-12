:mod:`kiwi.training.optimizers`
===============================

.. py:module:: kiwi.training.optimizers


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.training.optimizers.OptimizerConfig



Functions
~~~~~~~~~

.. autoapisummary::

   kiwi.training.optimizers.get_noam_decay_schedule
   kiwi.training.optimizers.optimizer_class
   kiwi.training.optimizers.optimizer_name
   kiwi.training.optimizers.from_config


.. data:: logger
   

   

.. py:class:: OptimizerConfig

   Bases: :class:`kiwi.utils.io.BaseConfig`

   Base class for all pydantic configs. Used to configure base behaviour of configs.

   .. attribute:: class_name
      :annotation: :str

      

   .. attribute:: learning_rate
      :annotation: :float

      Starting learning rate. Recommended settings:
      sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001


   .. attribute:: encoder_learning_rate
      :annotation: :float

      Different learning rate for the encoder. If set, the encoder will use a different
      learning rate from the rest of the parameters.


   .. attribute:: warmup_steps
      :annotation: :Union[float, int]

      Increase the learning rate until X steps. Integer steps for `noam` optimizer and
      `adamw`. If float, use it as portion of ``training_steps``.


   .. attribute:: training_steps
      :annotation: :int

      Total number of training steps. Used for the `adamw` optimizer. If not specified,
      use training dataset size.


   .. attribute:: learning_rate_decay
      :annotation: :float = 1.0

      ``new_lr = lr * factor``.
      Scheduler is only used if this is greater than 0.

      :type: Factor by which the learning rate will be reduced


   .. attribute:: learning_rate_decay_start
      :annotation: :int = 2

      Number of epochs with no improvement after which learning rate will be reduced.
      Only applicable if ``learning_rate_decay`` is greater than 0.


   .. attribute:: load
      :annotation: :Path

      

   .. method:: cast_steps(cls, v)



.. function:: get_noam_decay_schedule(optimizer: Optimizer, num_warmup_steps: int, model_size: int)

   Create a schedule with the learning rate decay strategy from the AIAYN paper.

   :param optimizer: wrapped optimizer.
   :param num_warmup_steps: the number of steps to linearly increase the learning rate.
   :param model_size: the hidden size parameter which dominates the number of
                      parameters in your model.


.. data:: OPTIMIZERS_MAPPING
   

   

.. function:: optimizer_class(name)


.. function:: optimizer_name(cls)


.. function:: from_config(config: OptimizerConfig, parameters: Iterator[Parameter], model_size: int = None, training_data_size: int = None) -> Union[Optimizer, List[Optimizer], Tuple[List[Optimizer], List[Any]]]

   :param config: common options shared by most optimizers
   :param parameters: model parameters
   :param model_size: required for the Noam LR schedule; if not provided, the mode of all
                      parameters' last dimension is used

   Return: for compatibility with PyTorch-Lightning, any of these 3 options:
       - Single optimizer
       - List or Tuple - List of optimizers
       - Tuple of Two lists - The first with multiple optimizers, the second with
                              learning-rate schedulers

   .. rubric:: Notes

   We currently never return multiple optimizers or schedulers, so option 2 above
   is not taking place yet. Option 3 returns a single optimizer and scheduler
   inside lists.


