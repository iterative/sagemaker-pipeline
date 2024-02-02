import argparse
import os

import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.xgboost.estimator import XGBoost


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
    parser.add_argument("--objective", type=str)
    parser.add_argument("--num_round", type=int)

    args = parser.parse_args()
    bucket = args.bucket
    prefix = args.prefix

    train_path = f"s3://{bucket}/{prefix}/train"
    validation_path = f"s3://{bucket}/{prefix}/validation"
    
    #container = sagemaker.image_uris.retrieve(region=boto3.Session().region_name, framework='xgboost', version='latest')

    s3_input_train = sagemaker.inputs.TrainingInput(s3_data=train_path.format(bucket, prefix), content_type='csv')
    s3_input_validation = sagemaker.inputs.TrainingInput(s3_data=validation_path.format(bucket, prefix), content_type='csv')
    
    #sess = sagemaker.Session()
    
    env = {name: value for name, value in os.environ.items() if name.startswith("DVC")}
    print(env)
    
    #xgb = sagemaker.estimator.Estimator(container,
    #                                    get_execution_role(),
    #                                    instance_count=1,
    #                                    instance_type='ml.m4.xlarge',
    #                                    output_path='s3://{}/{}/output'.format(bucket, prefix),
    #                                    sagemaker_session=sess,
    #                                    entry_point="train.py", # include training script
    #                                    source_dir=".", # include repo path that points to requirements.txt
    #                                    env=env, # pass dvc environment variables
    #                                   )
    
    #xgb.set_hyperparameters(max_depth=args.max_depth,
    #                        eta=args.eta,
    #                        gamma=args.gamma,
    #                        min_child_weight=args.min_child_weight,
    #                        subsample=args.subsample,
    #                        silent=args.silent,
    #                        objective=args.objective,
    #                        num_round=args.num_round)

    hyperparameters = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "gamma": args.gamma,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "silent": args.silent,
        #"objective": args.objective,
        "num_round": args.num_round,
    }    
    
    xgb = XGBoost(
        role=get_execution_role(),
        instance_count=1,
        instance_type="ml.m4.xlarge",
        framework_version="latest",
        output_path='s3://{}/{}/output'.format(bucket, prefix),
        hyperparameters=hyperparameters,
        entry_point="train.py", # include training script
        source_dir=".", # include repo path that points to requirements.txt
        env=env, # pass dvc environment variables
    )
    
    xgb.fit({'train': s3_input_train, 'validation': s3_input_validation}) 


if __name__ == "__main__":
    main()