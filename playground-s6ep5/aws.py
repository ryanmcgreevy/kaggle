import boto3
import sagemaker
from sagemaker.remote_function import remote
import tune
import argparse
from sagemaker import image_uris
from sagemaker.session import get_execution_role
from sagemaker.session import Session
import os
# Specify your desired region
region = 'us-east-1'  # Change this to your preferred region

# session = boto3.Session(region_name=region, profile_name='sagemaker')
# sm_session = sagemaker.Session(boto_session=session)
# role = "arn:aws:iam::801638386573:role/service-role/AmazonSageMaker-ExecutionRole-20260227T135383"



session = boto3.Session(region_name=region)  # no profile_name
sm_session = sagemaker.Session(boto_session=session)
role = "arn:aws:iam::801638386573:role/service-role/AmazonSageMaker-ExecutionRole-20260227T135383"

# uri = image_uris.retrieve(
#     framework='sagemaker-base-python',
#     region='us-east-1',
#     version='3.12',
#     instance_type='ml.m5.xlarge',
#     sagemaker_session=sm_session
# )
# print(uri)

settings = dict(
    sagemaker_session=sm_session,
    role=role,
    #instance_type="ml.g4dn.xlarge",
    instance_type="ml.m5.xlarge",
    dependencies='./requirements.txt',
    include_local_workdir=True,
    #image_uri="public.ecr.aws/deep-learning-containers/pytorch-training:2.9-gpu-py312-cu130-ubuntu22.04-sagemaker-v1"
    #image_uri=f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker"
    #image_uri="801638386573.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.9-gpu-py312-cu130-ubuntu22.04-sagemaker-v1"
    image_uri="801638386573.dkr.ecr.us-east-1.amazonaws.com/aws_test:latest"
)

@remote(**settings)
def tune_on_aws(classifier='lgbm'):
    print("Running on AWS SageMaker!")
    os.environ['MLFLOW_SERVER'] = "arn:aws:sagemaker:us-east-1:801638386573:mlflow-tracking-server/mlflow-test-server"
    tune.main(classifier)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tune.py on SageMaker Remote Function.")
    parser.add_argument(
        "--classifier",
        type=str,
        default="lgbm",
        choices=["lgbm", "cb", "hgb", "nn"],
        help="Classifier to tune: lgbm, cb, hgb, or nn",
    )
    args = parser.parse_args()
    tune_on_aws(args.classifier)