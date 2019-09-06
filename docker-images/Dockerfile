FROM 520713654638.dkr.ecr.ap-northeast-1.amazonaws.com/sagemaker-chainer:5.0.0-gpu-py3

RUN pip install -U setuptools pip
RUN pip uninstall -y cupy
RUN pip install 'cupy-cuda90>=6.3.0,<7.0.0'
RUN pip install -U chainer==6.3.0

