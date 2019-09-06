import argparse
import os

import sagemaker
from boto3.session import Session
from sagemaker import LocalSession
from sagemaker.chainer.estimator import Chainer


# See https://docs.aws.amazon.com/ja_jp/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html
IMAGE = '520713654638.dkr.ecr.ap-northeast-1.amazonaws.com/sagemaker-chainer:5.0.0-gpu-py3'

DEFAULT_INSTANCE_TYPE = 'ml.m4.4xlarge'
DEFAULT_REGION = 'ap-northeast-1'
DEFAULT_RUNTIME = 30  # seconds


def main(parser=argparse.ArgumentParser()):
    import logging
    logging.basicConfig(level=logging.WARN)

    parser.add_argument('--profile', type=str, default='default')
    parser.add_argument('--local-mode', action='store_true')
    parser.add_argument('--instance-type', type=str, default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument('--region', type=str, default=DEFAULT_REGION)
    parser.add_argument('--arn', type=str, default=None)
    parser.add_argument('--max-runtime', type=int, default=DEFAULT_RUNTIME, help='seconds')
    args = parser.parse_args()

    boto_session = Session(profile_name=args.profile, region_name=args.region)
    sagemaker_session = LocalSession(boto_session) if args.local_mode else sagemaker.Session(boto_session)
    role = args.arn if args.arn is not None else sagemaker.get_execution_role(sagemaker_session)

    gpu = 0 if args.instance_type.startswith('ml.p') and not args.local_mode else -1
    hyperparameters = {'gpu': gpu,
                       }
    chainer_estimator = Chainer(entry_point='train.py',
                                source_dir='./',
                                role=role,
                                image_name=IMAGE,
                                framework_version='5.0.0',
                                sagemaker_session=sagemaker_session,
                                train_instance_count=1,
                                train_instance_type='local' if args.local_mode else args.instance_type,
                                hyperparameters=hyperparameters,
                                base_job_name='chainer-sagemaker-sample',
                                train_max_run=args.max_runtime)
    chainer_estimator.fit(wait=args.local_mode)


if __name__ == '__main__':
    main()
