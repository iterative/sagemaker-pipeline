import argparse
import boto3
import sagemaker
from sagemaker import get_execution_role


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket")
    parser.add_argument("--prefix")
    parser.add_argument("--max_depth", type=int)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--gamma", type=int)
    parser.add_argument("--min_child_weight", type=int)
    parser.add_argument("--subsample", type=float)
    parser.add_argument("--silent", type=int)
    parser.add_argument("--objective")
    parser.add_argument("--num_round", type=int)

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
    xgb.set_hyperparameters(max_depth=args.max_depth,
                            eta=args.eta,
                            gamma=args.gamma,
                            min_child_weight=args.min_child_weight,
                            subsample=args.subsample,
                            silent=args.silent,
                            objective=args.objective,
                            num_round=args.num_round)

    xgb.fit({'train': s3_input_train, 'validation': s3_input_validation}) 


if __name__ == "__main__":
    main()