# syntax = docker/dockerfile:1.0-experimental
# Dockerfile for whatplane API
FROM python:3.8.7-slim-buster

ARG PYTHON_PACKAGES="fastapi==0.63.0 uvicorn[standard]==0.13.3 azure-storage-blob==12.7.1"
ARG APT_DEPS="dumb-init"

ARG APP_DIR="/whatplane"
ARG SCRIPTS_DIR=${APP_DIR}"/scripts/api"
ARG IMAGENET_MODEL="densenet161"
ARG WHATPLANE_MODEL="model.pth"
ARG MODELS_DIR=${APP_DIR}"/models"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ${APT_DEPS} \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    &&  pip install --no-cache-dir torch==1.7.1+cpu torchvision==0.8.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install --no-cache-dir ${PYTHON_PACKAGES}

COPY scripts/api ${SCRIPTS_DIR}

# Download required models into image
RUN mkdir -p ${MODELS_DIR} \
    && python ${SCRIPTS_DIR}/fetch_model_pytorch.py ${IMAGENET_MODEL} ${MODELS_DIR}

RUN --mount=type=secret,id=ENV \
    python ${SCRIPTS_DIR}/fetch_model_azure.py ${WHATPLANE_MODEL} ${MODELS_DIR}

COPY api ${APP_DIR}/api
COPY whatplane ${APP_DIR}/whatplane

RUN chmod a+x ${SCRIPTS_DIR}/entrypoint.sh

EXPOSE 5000

WORKDIR "${APP_DIR}"

ENTRYPOINT ["/usr/bin/dumb-init", "--", "/whatplane/scripts/api/entrypoint.sh"]

CMD ["start-uvicorn"]
