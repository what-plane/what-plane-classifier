# Dockerfile for whatplane application
FROM python:3.8.6-slim-buster

ARG PYTHON_PACKAGES="flask==1.1.2 matplotlib numpy tqdm scikit-learn tensorboard"

RUN pip install --upgrade pip

RUN pip install --user torch==1.7.0+cpu torchvision==0.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install ${PYTHON_PACKAGES}

COPY app app
COPY models/model_ash_densenet161_SGD.pth models/model_ash_densenet161_SGD.pth
COPY src src

EXPOSE 5000

CMD ["python", "app/app.py"]