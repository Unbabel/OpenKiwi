Systems
=======

There are two types of systems: **QE** and **TLM**. The first is the regular one used
for Quality Estimation. The second, which stands for *Translation Language Model*, is
used for pre-training the *encoder* component of a QE system.


All systems, regardless of them being **QE** or **TLM** systems are constructed on top
of PytorchLightning's (PTL) `LightningModule` class. This means they respect a certain design
philosophy that can be consulted in PTL's documentation_.

.. _documentation: https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html

Furthermore, all of our QE systems share a similar architecture. They are composed of

- Encoder
- Decoder
- Output
- (Optionally) TLM Output

TLM systems on the other hand are only composed of the Encoder + TLM Output and their
goal is to pre-train the encoder so it can be plugged into a QE system.

The systems are divided into the 3 blocks that have the major responsabilities in both
word and sentence level QE tasks:

Encoder: Embedding and creating features to be used for downstream tasks. i.e. Predictor,
BERT, etc

Decoder: Responsible for learning feature transformations better suited for the
downstream task. i.e. MLP, LSTM, etc

Output: Simple feedforwards that take decoder features and transform them into the
prediction required by the downstream task. Something in the same line as the common
"classification heads" being used with transformers.

TLM Output: A simple output layer that trains for the specific TLM objective. It can be
useful to continue finetuning the predictor during training of the complete QE system.


QE --- :mod:`kiwi.systems.qe_system`
------------------------------------

All QE systems inherit from :class:`kiwi.systems.qe_system.QESystem`.

Use ``kiwi train`` to train these systems.

Currently available are:

+--------------------------------------------------------------+
| :class:`kiwi.systems.nuqe.NuQE`                              |
+--------------------------------------------------------------+
| :class:`kiwi.systems.predictor_estimator.PredictorEstimator` |
+--------------------------------------------------------------+
| :class:`kiwi.systems.bert.Bert`                              |
+--------------------------------------------------------------+
| :class:`kiwi.systems.xlm.XLM`                                |
+--------------------------------------------------------------+
| :class:`kiwi.systems.xlmroberta.XLMRoberta`                  |
+--------------------------------------------------------------+


TLM --- :mod:`kiwi.systems.tlm_system`
--------------------------------------

All TLM systems inherit from :class:`kiwi.systems.tlm_system.TLMSystem`.

Use ``kiwi pretrain`` to train these systems. These systems can then be used as the
encoder part in QE systems by using the `load_encoder` flag.

Currently available are:

+-------------------------------------------+
| :class:`kiwi.systems.predictor.Predictor` |
+-------------------------------------------+
