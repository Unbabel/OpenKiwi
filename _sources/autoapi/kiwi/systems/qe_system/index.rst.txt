:mod:`kiwi.systems.qe_system`
=============================

.. py:module:: kiwi.systems.qe_system


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.systems.qe_system.BatchSizeConfig
   kiwi.systems.qe_system.ModelConfig
   kiwi.systems.qe_system.QESystem



.. data:: logger
   

   

.. py:class:: BatchSizeConfig

   Bases: :class:`kiwi.utils.io.BaseConfig`

   Base class for all pydantic configs. Used to configure base behaviour of configs.

   .. attribute:: train
      :annotation: :PositiveInt = 1

      

   .. attribute:: valid
      :annotation: :PositiveInt = 1

      

   .. attribute:: test
      :annotation: :PositiveInt = 1

      


.. py:class:: ModelConfig

   Bases: :class:`kiwi.utils.io.BaseConfig`

   Base class for all pydantic configs. Used to configure base behaviour of configs.

   .. attribute:: encoder
      :annotation: :Any

      

   .. attribute:: decoder
      :annotation: :Any

      

   .. attribute:: outputs
      :annotation: :Any

      

   .. attribute:: tlm_outputs
      :annotation: :Any

      


.. py:class:: QESystem(config, data_config: WMTQEDataset.Config = None)

   Bases: :class:`kiwi.systems._meta_module.Serializable`, :class:`pytorch_lightning.LightningModule`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:class:: Config

      Bases: :class:`kiwi.utils.io.BaseConfig`

      System configuration base class.

      .. attribute:: class_name
         :annotation: :Optional[str]

         System class to use (must be a subclass of ``QESystem`` and decorated with
         ``@QESystem.register_subclass``).


      .. attribute:: load
         :annotation: :Optional[Path]

         Load pretrained Kiwi model.
         If set, system architecture and vocabulary parameters are ignored.


      .. attribute:: load_encoder
         :annotation: :Optional[Path]

         Load pretrained encoder (e.g., the Predictor).
         If set, encoder architecture and vocabulary parameters are ignored
         (for the fields that are part of the encoder).


      .. attribute:: load_vocabs
         :annotation: :Optional[Path]

         

      .. attribute:: model
         :annotation: :Optional[Dict]

         System specific options; they will be dynamically validated and instantiated
         depending of the ``class_name`` or ``load``.


      .. attribute:: data_processing
         :annotation: :Optional[WMTQEDataEncoder.Config]

         

      .. attribute:: optimizer
         :annotation: :Optional[optimizers.OptimizerConfig]

         

      .. attribute:: batch_size
         :annotation: :BatchSizeConfig = 1

         

      .. attribute:: num_data_workers
         :annotation: :int = 4

         

      .. method:: map_name_to_class(cls, v)


      .. method:: check_consistency(cls, v, values)


      .. method:: check_model_requirement(cls, v, values)


      .. method:: check_batching(cls, v)



   .. attribute:: subclasses
      

      

   .. method:: _load_encoder(self, path: Path)


   .. method:: set_config_options(self, optimizer_config: optimizers.OptimizerConfig = None, batch_size: BatchSizeConfig = None, num_data_workers: int = None, data_config: WMTQEDataset.Config = None)


   .. method:: prepare_data(self)

      Initialize the data sources the model will use to create the data loaders.


   .. method:: train_dataloader(self) -> torch.utils.data.DataLoader

      Return a PyTorch DataLoader for the training set.

      Requires calling ``prepare_data`` beforehand.

      :returns: PyTorch DataLoader


   .. method:: val_dataloader(self) -> torch.utils.data.DataLoader

      Return a PyTorch DataLoader for the validation set.

      Requires calling ``prepare_data`` beforehand.

      :returns: PyTorch DataLoader


   .. method:: test_dataloader(self) -> Optional[DataLoader]

      Implement one or multiple PyTorch DataLoaders for testing.

      The dataloader you return will not be called every epoch unless you set
      :paramref:`~pytorch_lightning.trainer.Trainer.reload_dataloaders_every_epoch` to ``True``.

      For data processing use the following pattern:

          - download in :meth:`prepare_data`
          - process and split in :meth:`setup`

      However, the above are only necessary for distributed processing.

      .. warning:: do not assign state in prepare_data


      - :meth:`~pytorch_lightning.trainer.Trainer.fit`
      - ...
      - :meth:`prepare_data`
      - :meth:`setup`
      - :meth:`train_dataloader`
      - :meth:`val_dataloader`
      - :meth:`test_dataloader`

      .. note::

         Lightning adds the correct sampler for distributed and arbitrary hardware.
         There is no need to set it yourself.

      :returns: Single or multiple PyTorch DataLoaders.

      .. rubric:: Example

      .. code-block:: python

          def test_dataloader(self):
              transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5,), (1.0,))])
              dataset = MNIST(root='/path/to/mnist/', train=False, transform=transform,
                              download=True)
              loader = torch.utils.data.DataLoader(
                  dataset=dataset,
                  batch_size=self.batch_size,
                  shuffle=False
              )

              return loader

      .. note::

         If you don't need a test dataset and a :meth:`test_step`, you don't need to implement
         this method.


   .. method:: prepare_dataloader(self, dataset: WMTQEDataset, batch_size: int = 1, num_workers: int = 0)


   .. method:: forward(self, batch_inputs)

      Same as :meth:`torch.nn.Module.forward()`, however in Lightning you want this to define
      the operations you want to use for prediction (i.e.: on a server or as a feature extractor).

      Normally you'd call ``self()`` from your :meth:`training_step` method.
      This makes it easy to write a complex system for training with the outputs
      you'd want in a prediction setting.

      You may also find the :func:`~pytorch_lightning.core.decorators.auto_move_data` decorator useful
      when using the module outside Lightning in a production setting.

      :param \*args: Whatever you decide to pass into the forward method.
      :param \*\*kwargs: Keyword arguments are also possible.

      :returns: Predicted output

      .. rubric:: Examples

      .. code-block:: python

          # example if we were using this model as a feature extractor
          def forward(self, x):
              feature_maps = self.convnet(x)
              return feature_maps

          def training_step(self, batch, batch_idx):
              x, y = batch
              feature_maps = self(x)
              logits = self.classifier(feature_maps)

              # ...
              return loss

          # splitting it this way allows model to be used a feature extractor
          model = MyModelAbove()

          inputs = server.get_request()
          results = model(inputs)
          server.write_results(results)

          # -------------
          # This is in stark contrast to torch.nn.Module where normally you would have this:
          def forward(self, batch):
              x, y = batch
              feature_maps = self.convnet(x)
              logits = self.classifier(feature_maps)
              return logits


   .. method:: training_step(self, batch: MultiFieldBatch, batch_idx: int) -> Dict[str, Dict[str, Tensor]]

      Here you compute and return the training loss and some additional metrics for e.g.
      the progress bar or logger.

      :param batch: The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
      :type batch: :class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]
      :param batch_idx: Integer displaying index of this batch
      :type batch_idx: int
      :param optimizer_idx: When using multiple optimizers, this argument will also be present.
      :type optimizer_idx: int
      :param hiddens: Passed in if
                      :paramref:`~pytorch_lightning.trainer.trainer.Trainer.truncated_bptt_steps` > 0.
      :type hiddens: :class:`~torch.Tensor`

      :returns: Dict with loss key and optional log or progress bar keys.
                When implementing :meth:`training_step`, return whatever you need in that step:

                - loss -> tensor scalar **REQUIRED**
                - progress_bar -> Dict for progress bar display. Must have only tensors
                - log -> Dict of metrics to add to logger. Must have only tensors (no images, etc)

      In this step you'd normally do the forward pass and calculate the loss for a batch.
      You can also do fancier things like multiple forward passes or something model specific.

      .. rubric:: Examples

      .. code-block:: python

          def training_step(self, batch, batch_idx):
              x, y, z = batch

              # implement your own
              out = self(x)
              loss = self.loss(out, x)

              logger_logs = {'training_loss': loss} # optional (MUST ALL BE TENSORS)

              # if using TestTubeLogger or TensorBoardLogger you can nest scalars
              logger_logs = {'losses': logger_logs} # optional (MUST ALL BE TENSORS)

              output = {
                  'loss': loss, # required
                  'progress_bar': {'training_loss': loss}, # optional (MUST ALL BE TENSORS)
                  'log': logger_logs
              }

              # return a dict
              return output

      If you define multiple optimizers, this step will be called with an additional
      ``optimizer_idx`` parameter.

      .. code-block:: python

          # Multiple optimizers (e.g.: GANs)
          def training_step(self, batch, batch_idx, optimizer_idx):
              if optimizer_idx == 0:
                  # do training_step with encoder
              if optimizer_idx == 1:
                  # do training_step with decoder


      If you add truncated back propagation through time you will also get an additional
      argument with the hidden states of the previous step.

      .. code-block:: python

          # Truncated back-propagation through time
          def training_step(self, batch, batch_idx, hiddens):
              # hiddens are the hidden states from the previous truncated backprop step
              ...
              out, hiddens = self.lstm(data, hiddens)
              ...

              return {
                  "loss": ...,
                  "hiddens": hiddens  # remember to detach() this
              }

      .. rubric:: Notes

      The loss value shown in the progress bar is smoothed (averaged) over the last values,
      so it differs from the actual loss returned in train/validation step.


   .. method:: training_epoch_end(self, outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]) -> Dict[str, Union[torch.Tensor, Dict[str, Tensor]]]

      Called at the end of the training epoch with the outputs of all training steps.

      .. code-block:: python

          # the pseudocode for these calls
          train_outs = []
          for train_batch in train_data:
              out = training_step(train_batch)
              train_outs.append(out)
          training_epoch_end(train_outs)

      :param outputs: List of outputs you defined in :meth:`training_step`, or if there are
                      multiple dataloaders, a list containing a list of outputs for each dataloader.

      :returns: Dict or OrderedDict.
                May contain the following optional keys:

                - log (metrics to be added to the logger; only tensors)
                - progress_bar (dict for progress bar display)
                - any metric used in a callback (e.g. early stopping).

      .. note:: If this method is not overridden, this won't be called.

      - The outputs here are strictly for logging or progress bar.
      - If you don't need to display anything, don't return anything.
      - If you want to manually set current step, you can specify the 'step' key in the 'log' dict.

      .. rubric:: Examples

      With a single dataloader:

      .. code-block:: python

          def training_epoch_end(self, outputs):
              train_acc_mean = 0
              for output in outputs:
                  train_acc_mean += output['train_acc']

              train_acc_mean /= len(outputs)

              # log training accuracy at the end of an epoch
              results = {
                  'log': {'train_acc': train_acc_mean.item()},
                  'progress_bar': {'train_acc': train_acc_mean},
              }
              return results

      With multiple dataloaders, ``outputs`` will be a list of lists. The outer list contains
      one entry per dataloader, while the inner list contains the individual outputs of
      each training step for that dataloader.

      .. code-block:: python

          def training_epoch_end(self, outputs):
              train_acc_mean = 0
              i = 0
              for dataloader_outputs in outputs:
                  for output in dataloader_outputs:
                      train_acc_mean += output['train_acc']
                      i += 1

              train_acc_mean /= i

              # log training accuracy at the end of an epoch
              results = {
                  'log': {'train_acc': train_acc_mean.item(), 'step': self.current_epoch}
                  'progress_bar': {'train_acc': train_acc_mean},
              }
              return results


   .. method:: validation_step(self, batch, batch_idx)

      Operates on a single batch of data from the validation set.
      In this step you'd might generate examples or calculate anything of interest like accuracy.

      .. code-block:: python

          # the pseudocode for these calls
          val_outs = []
          for val_batch in val_data:
              out = validation_step(train_batch)
              val_outs.append(out)
              validation_epoch_end(val_outs)

      :param batch: The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
      :type batch: :class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]
      :param batch_idx: The index of this batch
      :type batch_idx: int
      :param dataloader_idx: The index of the dataloader that produced this batch
                             (only if multiple val datasets used)
      :type dataloader_idx: int

      :returns: Dict or OrderedDict - passed to :meth:`validation_epoch_end`.
                If you defined :meth:`validation_step_end` it will go to that first.

      .. code-block:: python

          # pseudocode of order
          out = validation_step()
          if defined('validation_step_end'):
              out = validation_step_end(out)
          out = validation_epoch_end(out)


      .. code-block:: python

          # if you have one val dataloader:
          def validation_step(self, batch, batch_idx)

          # if you have multiple val dataloaders:
          def validation_step(self, batch, batch_idx, dataloader_idx)

      .. rubric:: Examples

      .. code-block:: python

          # CASE 1: A single validation dataset
          def validation_step(self, batch, batch_idx):
              x, y = batch

              # implement your own
              out = self(x)
              loss = self.loss(out, y)

              # log 6 example images
              # or generated text... or whatever
              sample_imgs = x[:6]
              grid = torchvision.utils.make_grid(sample_imgs)
              self.logger.experiment.add_image('example_images', grid, 0)

              # calculate acc
              labels_hat = torch.argmax(out, dim=1)
              val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

              # all optional...
              # return whatever you need for the collation function validation_epoch_end
              output = OrderedDict({
                  'val_loss': loss_val,
                  'val_acc': torch.tensor(val_acc), # everything must be a tensor
              })

              # return an optional dict
              return output

      If you pass in multiple val datasets, validation_step will have an additional argument.

      .. code-block:: python

          # CASE 2: multiple validation datasets
          def validation_step(self, batch, batch_idx, dataset_idx):
              # dataset_idx tells you which dataset this is.

      .. note:: If you don't need to validate you don't need to implement this method.

      .. note::

         When the :meth:`validation_step` is called, the model has been put in eval mode
         and PyTorch gradients have been disabled. At the end of validation,
         the model goes back to training mode and gradients are enabled.


   .. method:: validation_epoch_end(self, outputs: List[Dict[str, Dict[str, Tensor]]]) -> Dict[str, Dict[str, Tensor]]

      Called at the end of the validation epoch with the outputs of all validation steps.

      .. code-block:: python

          # the pseudocode for these calls
          val_outs = []
          for val_batch in val_data:
              out = validation_step(val_batch)
              val_outs.append(out)
          validation_epoch_end(val_outs)

      :param outputs: List of outputs you defined in :meth:`validation_step`, or if there
                      are multiple dataloaders, a list containing a list of outputs for each dataloader.

      :returns: Dict or OrderedDict.
                May have the following optional keys:

                - progress_bar (dict for progress bar display; only tensors)
                - log (dict of metrics to add to logger; only tensors).

      .. note:: If you didn't define a :meth:`validation_step`, this won't be called.

      - The outputs here are strictly for logging or progress bar.
      - If you don't need to display anything, don't return anything.
      - If you want to manually set current step, you can specify the 'step' key in the 'log' dict.

      .. rubric:: Examples

      With a single dataloader:

      .. code-block:: python

          def validation_epoch_end(self, outputs):
              val_acc_mean = 0
              for output in outputs:
                  val_acc_mean += output['val_acc']

              val_acc_mean /= len(outputs)
              tqdm_dict = {'val_acc': val_acc_mean.item()}

              # show val_acc in progress bar but only log val_loss
              results = {
                  'progress_bar': tqdm_dict,
                  'log': {'val_acc': val_acc_mean.item()}
              }
              return results

      With multiple dataloaders, `outputs` will be a list of lists. The outer list contains
      one entry per dataloader, while the inner list contains the individual outputs of
      each validation step for that dataloader.

      .. code-block:: python

          def validation_epoch_end(self, outputs):
              val_acc_mean = 0
              i = 0
              for dataloader_outputs in outputs:
                  for output in dataloader_outputs:
                      val_acc_mean += output['val_acc']
                      i += 1

              val_acc_mean /= i
              tqdm_dict = {'val_acc': val_acc_mean.item()}

              # show val_loss and val_acc in progress bar but only log val_loss
              results = {
                  'progress_bar': tqdm_dict,
                  'log': {'val_acc': val_acc_mean.item(), 'step': self.current_epoch}
              }
              return results


   .. method:: test_step(self, *args, **kwargs) -> Dict[str, Tensor]

      Operates on a single batch of data from the test set.
      In this step you'd normally generate examples or calculate anything of interest
      such as accuracy.

      .. code-block:: python

          # the pseudocode for these calls
          test_outs = []
          for test_batch in test_data:
              out = test_step(test_batch)
              test_outs.append(out)
          test_epoch_end(test_outs)

      :param batch: The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
      :type batch: :class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]
      :param batch_idx: The index of this batch.
      :type batch_idx: int
      :param dataloader_idx: The index of the dataloader that produced this batch
                             (only if multiple test datasets used).
      :type dataloader_idx: int

      :returns: Dict or OrderedDict - passed to the :meth:`test_epoch_end` method.
                If you defined :meth:`test_step_end` it will go to that first.

      .. code-block:: python

          # if you have one test dataloader:
          def test_step(self, batch, batch_idx)

          # if you have multiple test dataloaders:
          def test_step(self, batch, batch_idx, dataloader_idx)

      .. rubric:: Examples

      .. code-block:: python

          # CASE 1: A single test dataset
          def test_step(self, batch, batch_idx):
              x, y = batch

              # implement your own
              out = self(x)
              loss = self.loss(out, y)

              # log 6 example images
              # or generated text... or whatever
              sample_imgs = x[:6]
              grid = torchvision.utils.make_grid(sample_imgs)
              self.logger.experiment.add_image('example_images', grid, 0)

              # calculate acc
              labels_hat = torch.argmax(out, dim=1)
              val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

              # all optional...
              # return whatever you need for the collation function test_epoch_end
              output = OrderedDict({
                  'val_loss': loss_val,
                  'val_acc': torch.tensor(val_acc), # everything must be a tensor
              })

              # return an optional dict
              return output

      If you pass in multiple validation datasets, :meth:`test_step` will have an additional
      argument.

      .. code-block:: python

          # CASE 2: multiple test datasets
          def test_step(self, batch, batch_idx, dataset_idx):
              # dataset_idx tells you which dataset this is.

      .. note:: If you don't need to validate you don't need to implement this method.

      .. note::

         When the :meth:`test_step` is called, the model has been put in eval mode and
         PyTorch gradients have been disabled. At the end of the test epoch, the model goes back
         to training mode and gradients are enabled.


   .. method:: test_epoch_end(self, outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]) -> Dict[str, Dict[str, Tensor]]

      Called at the end of a test epoch with the output of all test steps.

      .. code-block:: python

          # the pseudocode for these calls
          test_outs = []
          for test_batch in test_data:
              out = test_step(test_batch)
              test_outs.append(out)
          test_epoch_end(test_outs)

      :param outputs: List of outputs you defined in :meth:`test_step_end`, or if there
                      are multiple dataloaders, a list containing a list of outputs for each dataloader

      :returns: Dict has the following optional keys:

                - progress_bar -> Dict for progress bar display. Must have only tensors.
                - log -> Dict of metrics to add to logger. Must have only tensors (no images, etc).
      :rtype: Dict or OrderedDict

      .. note:: If you didn't define a :meth:`test_step`, this won't be called.

      - The outputs here are strictly for logging or progress bar.
      - If you don't need to display anything, don't return anything.
      - If you want to manually set current step, specify it with the 'step' key in the 'log' Dict

      .. rubric:: Examples

      With a single dataloader:

      .. code-block:: python

          def test_epoch_end(self, outputs):
              test_acc_mean = 0
              for output in outputs:
                  test_acc_mean += output['test_acc']

              test_acc_mean /= len(outputs)
              tqdm_dict = {'test_acc': test_acc_mean.item()}

              # show test_loss and test_acc in progress bar but only log test_loss
              results = {
                  'progress_bar': tqdm_dict,
                  'log': {'test_acc': test_acc_mean.item()}
              }
              return results

      With multiple dataloaders, `outputs` will be a list of lists. The outer list contains
      one entry per dataloader, while the inner list contains the individual outputs of
      each test step for that dataloader.

      .. code-block:: python

          def test_epoch_end(self, outputs):
              test_acc_mean = 0
              i = 0
              for dataloader_outputs in outputs:
                  for output in dataloader_outputs:
                      test_acc_mean += output['test_acc']
                      i += 1

              test_acc_mean /= i
              tqdm_dict = {'test_acc': test_acc_mean.item()}

              # show test_loss and test_acc in progress bar but only log test_loss
              results = {
                  'progress_bar': tqdm_dict,
                  'log': {'test_acc': test_acc_mean.item(), 'step': self.current_epoch}
              }
              return results


   .. method:: loss(self, model_out, batch)


   .. method:: metrics_step(self, batch, model_out, loss_dict)


   .. method:: metrics_end(self, steps, prefix='')


   .. method:: main_metric(self, selected_metric: Union[str, List[str]] = None) -> (Union[str, List[str]], str)

      Configure and retrieve the metric to be used for monitoring.

      The first time it is called, the main metric is configured based on the
      specified metrics in ``selected_metric`` or, if not provided, on the first
      metric in the outputs. Subsequent calls return the configured main metric.
      If a subsequent call specifies ``selected_metric``, configuration is done again.

      :returns:

                a tuple containing the main metric name and the ordering.
                    Note that the first element might be a concatenation of several
                    metrics in case ``selected_metric`` is a list. This is useful for
                    considering more than one metric as the best
                    (``metric_end()`` will sum over them).


   .. method:: num_parameters(self)


   .. method:: from_config(config: Config, data_config: WMTQEDataset.Config = None)
      :staticmethod:


   .. method:: load(cls, path: Path, map_location=None)
      :classmethod:


   .. method:: from_dict(cls, module_dict: Dict[str, Any])
      :classmethod:


   .. method:: _load_dict(self, module_dict)


   .. method:: to_dict(self, include_state=True)


   .. method:: on_save_checkpoint(self, checkpoint)

      Called by Lightning when saving a checkpoint to give you a chance to store anything
      else you might want to save.

      :param checkpoint: Checkpoint to be saved

      .. rubric:: Example

      .. code-block:: python


          def on_save_checkpoint(self, checkpoint):
              # 99% of use cases you don't need to implement this method
              checkpoint['something_cool_i_want_to_save'] = my_cool_pickable_object

      .. note::

         Lightning saves all aspects of training (epoch, global step, etc...)
         including amp scaling.
         There is no need for you to store anything about training.


   .. method:: on_load_checkpoint(self, checkpoint)

      Called by Lightning to restore your model.
      If you saved something with :meth:`on_save_checkpoint` this is your chance to restore this.

      :param checkpoint: Loaded checkpoint

      .. rubric:: Example

      .. code-block:: python

          def on_load_checkpoint(self, checkpoint):
              # 99% of the time you don't need to implement this method
              self.something_cool_i_want_to_save = checkpoint['something_cool_i_want_to_save']

      .. note::

         Lightning auto-restores global step, epoch, and train state including amp scaling.
         There is no need for you to restore anything regarding training.


   .. method:: load_from_checkpoint(cls, checkpoint_path: str, map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None, tags_csv: Optional[str] = None, *args, **kwargs) -> 'pl.LightningModule'
      :classmethod:

      Primary way of loading a model from a checkpoint. When Lightning saves a checkpoint
      it stores the arguments passed to `__init__`  in the checkpoint under `module_arguments`

      Any arguments specified through \*args and \*\*kwargs will override args stored in `hparams`.

      :param checkpoint_path: Path to checkpoint. This can also be a URL.
      :param args: Any positional args needed to init the model.
      :param map_location: If your checkpoint saved a GPU model and you now load on CPUs
                           or a different number of GPUs, use this to map to the new setup.
                           The behaviour is the same as in :func:`torch.load`.
      :param hparams_file: Optional path to a .yaml file with hierarchical structure
                           as in this example::

                               drop_prob: 0.2
                               dataloader:
                                   batch_size: 32

                           You most likely won't need this since Lightning will always save the hyperparameters
                           to the checkpoint.
                           However, if your checkpoint weights don't have the hyperparameters saved,
                           use this method to pass in a .yaml file with the hparams you'd like to use.
                           These will be converted into a :class:`~dict` and passed into your
                           :class:`LightningModule` for use.

                           If your model's `hparams` argument is :class:`~argparse.Namespace`
                           and .yaml file has hierarchical structure, you need to refactor your model to treat
                           `hparams` as :class:`~dict`.

                           .csv files are acceptable here till v0.9.0, see tags_csv argument for detailed usage.
      :param tags_csv:
                       .. warning:: .. deprecated:: 0.7.6

                           `tags_csv` argument is deprecated in v0.7.6. Will be removed v0.9.0.

                       Optional path to a .csv file with two columns (key, value)
                       as in this example::

                           key,value
                           drop_prob,0.2
                           batch_size,32

                       Use this method to pass in a .csv file with the hparams you'd like to use.
      :param hparam_overrides: A dictionary with keys to override in the hparams
      :param kwargs: Any keyword args needed to init the model.

      :returns: :class:`LightningModule` with loaded weights and hyperparameters (if available).

      .. rubric:: Example

      .. code-block:: python

          # load weights without mapping ...
          MyLightningModule.load_from_checkpoint('path/to/checkpoint.ckpt')

          # or load weights mapping all weights from GPU 1 to GPU 0 ...
          map_location = {'cuda:1':'cuda:0'}
          MyLightningModule.load_from_checkpoint(
              'path/to/checkpoint.ckpt',
              map_location=map_location
          )

          # or load weights and hyperparameters from separate files.
          MyLightningModule.load_from_checkpoint(
              'path/to/checkpoint.ckpt',
              hparams_file='/path/to/hparams_file.yaml'
          )

          # override some of the params with new values
          MyLightningModule.load_from_checkpoint(
              PATH,
              num_layers=128,
              pretrained_ckpt_path: NEW_PATH,
          )

          # predict
          pretrained_model.eval()
          pretrained_model.freeze()
          y_hat = pretrained_model(x)


   .. method:: configure_optimizers(self) -> Optional[Union[Optimizer, Sequence[Optimizer], Tuple[List, List]]]

      Instantiate configured optimizer and LR scheduler.

      Return: for compatibility with PyTorch-Lightning, any of these 3 options:
          - Single optimizer
          - List or Tuple - List of optimizers
          - Tuple of Two lists - The first with multiple optimizers, the second with
                                 learning-rate schedulers


   .. method:: predict(self, batch_inputs, positive_class_label=const.BAD)



