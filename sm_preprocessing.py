import argparse
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker import get_execution_role


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket")
    parser.add_argument("--prefix")
    args = parser.parse_args()
    bucket = args.bucket
    prefix = args.prefix

    input_source = f"s3://{bucket}/{prefix}/input_data"
    train_path = f"s3://{bucket}/{prefix}/train"
    validation_path = f"s3://{bucket}/{prefix}/validation"
    test_path = f"s3://{bucket}/{prefix}/test"


    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        role=get_execution_role(),
        instance_type="ml.m5.large",
        instance_count=1, 
        base_job_name='sm-immday-skprocessing'
    )


    sklearn_processor.run(
        code='preprocessing.py',
        # arguments = ['arg1', 'arg2'],
        inputs=[
            ProcessingInput(
                source=input_source, 
                destination="/opt/ml/processing/input",
                s3_input_mode="File",
                s3_data_distribution_type="ShardedByS3Key"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train_data", 
                source="/opt/ml/processing/output/train",
                destination=train_path,
            ),
            ProcessingOutput(output_name="validation_data", source="/opt/ml/processing/output/validation", destination=validation_path),
            ProcessingOutput(output_name="test_data", source="/opt/ml/processing/output/test", destination=test_path),
        ]
    )


if __name__ == "__main__":
    main()