import argparse
import pandas as pd
import zipfile
from sagemaker import Session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket")
    parser.add_argument("--prefix")
    args = parser.parse_args()
    bucket = args.bucket
    prefix = args.prefix

    with zipfile.ZipFile('bank-additional.zip', 'r') as zip_ref:
        zip_ref.extractall('.')    

    data = pd.read_csv('./bank-additional/bank-additional-full.csv')

    sess = Session()
    input_source = sess.upload_data('./bank-additional/bank-additional-full.csv', bucket=bucket, key_prefix=f'{prefix}/input_data')



if __name__ == "__main__":
    main()