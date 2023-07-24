import argparse
import boto3
import sagemaker
from sagemaker import get_execution_role


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket")
    parser.add_argument("--prefix")
    args = parser.parse_args()
    bucket = args.bucket
    prefix = args.prefix

    train_path = f"s3://{bucket}/{prefix}/train"
    validation_path = f"s3://{bucket}/{prefix}/validation"


    container = sagemaker.image_uris.retrieve(region=boto3.Session().region_name, framework='xgboost', version='latest')


    s3_input_train = sagemaker.inputs.TrainingInput(s3_data=train_path.format(bucket, prefix), content_type='csv')
    s3_input_validation = sagemaker.inputs.TrainingInput(s3_data=validation_path.format(bucket, prefix), content_type='csv')


    sess = sagemaker.Session()

    xgb = sagemaker.estimator.Estimator(container,
                                        get_execution_role(),
                                        instance_count=1, 
                                        instance_type='ml.m4.xlarge',
                                        output_path='s3://{}/{}/output'.format(bucket, prefix),
                                        sagemaker_session=sess)
    xgb.set_hyperparameters(max_depth=5,
                            eta=0.2,
                            gamma=4,
                            min_child_weight=6,
                            subsample=0.8,
                            silent=0,
                            objective='binary:logistic',
                            num_round=100)

    xgb.fit({'train': s3_input_train, 'validation': s3_input_validation}) 


if __name__ == "__main__":
    main()