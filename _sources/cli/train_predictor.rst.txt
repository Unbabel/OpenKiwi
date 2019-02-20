Predictor training
==================

This is used to pre-train the predictor side of the predictor-estimator model.

.. contents:: Contents
   :local:

.. argparse::
   :module: kiwi.cli.models.predictor_estimator
   :passparser:
   :func: add_pretraining_options
   :prog: kiwi train
   
   PredEst data
      --extend-source-vocab 
         This is useful to reduce OOV words if the parallel data and QE data are
         \from different domains.

