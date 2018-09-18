# AI &amp; Machine Learning for computer vision

Welcome to the AWS immersion day lab environment.

## Pre-requisite

We will be running the labs in the `us-east-1` (N. Virginia) region.

Please ensure the [SageMaker](./sagemaker-lab.json) stack is created, if not run it manually:

```
aws cloudformation create-stack \
  --region us-east-1 \
  --stack-name SageMaker \
  --template-body file://sagemaker-lab.json \
  --capabilities CAPABILITY_NAMED_IAM
```

This stack will create:

* A `SageMaker-LabExecutionRole` for running your SageMaker notebooks
* An Sagemaker S3 bucket of the name `sagemaker-{Region}-{AccountId}`

## Labs

Each lab has a read-me to get started.

* [Lab 1](lab1/Readme.md) AI Services
* [Lab 2](lab2/Readme.md) Introduction to Amazon SageMaker
* [Lab 3](lab3/Readme.md) Face Recognition
