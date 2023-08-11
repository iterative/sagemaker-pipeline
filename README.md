# sagemaker-pipeline

This repo takes the data processing and model training from https://github.com/aws-samples/amazon-sagemaker-immersion-day/blob/master/processing_xgboost.ipynb and converts it into a DVC pipeline. The code is minimally modified from the original notebook to modularize it into individual scripts and parametrize the s3 paths. To run it, modify the bucket and prefix paths in `params.yaml` and then use `dvc repro` or `dvc exp run` to execute the pipeline in SageMaker.

The pipeline has three stages:

1. Prepare data from S3
2. Run a preprocessing job using the [Scikit Learn Processor](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html#scikit-learn-processor)
3. Run a model training job using [XGBoost](https://sagemaker.readthedocs.io/en/stable/frameworks/xgboost/using_xgboost.html)

