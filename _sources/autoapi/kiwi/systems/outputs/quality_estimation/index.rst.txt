:mod:`kiwi.systems.outputs.quality_estimation`
==============================================

.. py:module:: kiwi.systems.outputs.quality_estimation


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.systems.outputs.quality_estimation.WordLevelConfig
   kiwi.systems.outputs.quality_estimation.SentenceLevelConfig
   kiwi.systems.outputs.quality_estimation.QEOutputs



Functions
~~~~~~~~~

.. autoapisummary::

   kiwi.systems.outputs.quality_estimation.tag_metrics


.. data:: logger
   

   

.. py:class:: WordLevelConfig

   Bases: :class:`kiwi.utils.io.BaseConfig`

   Base class for all pydantic configs. Used to configure base behaviour of configs.

   .. attribute:: target
      :annotation: :bool = False

      Train or predict target tags


   .. attribute:: gaps
      :annotation: :bool = False

      Train or predict gap tags


   .. attribute:: source
      :annotation: :bool = False

      Train or predict source tags


   .. attribute:: class_weights
      :annotation: :Dict[str, Dict[str, float]]

      Relative weight for labels on each output side.



.. py:class:: SentenceLevelConfig

   Bases: :class:`kiwi.utils.io.BaseConfig`

   Base class for all pydantic configs. Used to configure base behaviour of configs.

   .. attribute:: hter
      :annotation: :bool = False

      Predict Sentence Level Scores.
      Requires the appropriate input files (usually with HTER).


   .. attribute:: use_distribution
      :annotation: :bool = False

      Use probabilistic Loss for sentence scores instead of squared error.
      If set (requires `hter` to also be set), the model will output mean and variance
      of a truncated Gaussian distribution over the interval [0, 1], and use the NLL
      of ground truth scores as the loss.
      This seems to improve performance, and gives you uncertainty
      estimates for sentence level predictions as a byproduct.


   .. attribute:: binary
      :annotation: :bool = False

      Predict Binary Label for each sentence, indicating hter == 0.0.
      Requires the appropriate input files (usually with HTER).



.. py:class:: QEOutputs(inputs_dims, vocabs: Dict[str, Vocabulary], config: Config)

   Bases: :class:`kiwi.systems._meta_module.MetaModule`

   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super(Model, self).__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. py:class:: Config

      Bases: :class:`kiwi.utils.io.BaseConfig`

      Base class for all pydantic configs. Used to configure base behaviour of configs.

      .. attribute:: word_level
         :annotation: :WordLevelConfig

         

      .. attribute:: sentence_level
         :annotation: :SentenceLevelConfig

         

      .. attribute:: sentence_loss_weight
         :annotation: :float = 1.0

         Multiplier for sentence_level loss weight.


      .. attribute:: dropout
         :annotation: :float = 0.0

         

      .. attribute:: last_activation
         :annotation: :bool = False

         

      .. attribute:: n_layers_output
         :annotation: :int = 3

         


   .. method:: forward(self, features: Dict[str, Tensor], batch_inputs: MultiFieldBatch) -> Dict[str, Tensor]


   .. method:: loss(self, model_out: Dict[str, Tensor], batch: MultiFieldBatch) -> Dict[str, Tensor]


   .. method:: word_losses(self, model_out: Dict[str, Tensor], batch_outputs: MultiFieldBatch)

      Compute sequence tagging loss.


   .. method:: sentence_losses(self, model_out: Dict[str, Tensor], batch_outputs: MultiFieldBatch)

      Compute sentence score loss.


   .. method:: metrics_step(self, batch: MultiFieldBatch, model_out: Dict[str, Tensor], loss_dict: Dict[str, Tensor]) -> Dict[str, Tensor]


   .. method:: metrics_end(self, steps: List[Dict[str, Tensor]], prefix='')


   .. method:: metrics(self) -> List[Metric]
      :property:


   .. method:: labels(self, field: str) -> List[str]


   .. method:: decode_outputs(self, model_out: Dict[str, Tensor], batch_inputs: MultiFieldBatch, positive_class_label: str = const.BAD) -> Dict[str, List]


   .. method:: decode_word_outputs(self, model_out: Dict[str, Tensor], batch_inputs: MultiFieldBatch, positive_class_label: str = const.BAD) -> Dict[str, List]


   .. method:: decode_sentence_outputs(model_out: Dict[str, Tensor]) -> Dict[str, List]
      :staticmethod:



.. function:: tag_metrics(*targets, prefix=None, labels=None)


