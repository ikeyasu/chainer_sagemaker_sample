# chainer_sagemaker_sample

## Getting started

### 1. login to Docker

First of all, you need to login docker with AWS ECR account.

```bash
$ $(aws ecr get-login --no-include-email --region ap-northeast-1 --registry-ids 520713654638)
```

See
https://docs.aws.amazon.com/ja_jp/AmazonECR/latest/userguide/ECR_AWSCLI.html
for details.


`520713654638` is public ECR of SageMaker containers.

### 2. Get IAM role and dataset

1. Open SageMaker and Jupyter notebook
2. Upload `create_data_on_sagemaker.ipynb` to the notebook and open it.
3. Run all
4. Note IAM role. It is like `arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20181212T121212`
5. Note s3 path of dataset like `s3://sagemaker-ap-northeast-1-111111111111/notebook/chainer_cifar/train` and `s3://sagemaker-ap-northeast-1-111111111111/notebook/chainer_cifar/test`.

### 3. Local run

Make directory for SageMaker local.

```bash
$ mkdir -p ~/robo/container
```

Download `train_m.npz` and `test_m.npz`. Please use the noted s3 paths above.

```bash
$ aws s3 cp s3://sagemaker-ap-northeast-1-111111111111/notebook/chainer_cifar/train/train_m.npz train.npz
$ aws s3 cp s3://sagemaker-ap-northeast-1-111111111111/notebook/chainer_cifar/test/test_m.npz test.npz
```

Run SageMaker local mode. Please use the IAM role above.

```bash
$ python start_train.py --local-mode --profile ated --arn arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20181212T121212
```

## Custom docker

Login to your ECR.

```
 $(aws ecr get-login --no-include-email --region ap-northeast-1)
```

Build docker image.

```bash
$ cd docker-images
$ docker build -t 111111111111.dkr.ecr.ap-northeast-1.amazonaws.com/sagemaker-chainer:6.3.0-gpu-py3 .
```

Push to ECR.

```bash
$ docker push 111111111111.dkr.ecr.ap-northeast-1.amazonaws.com/sagemaker-chainer:6.3.0-gpu-py3
```