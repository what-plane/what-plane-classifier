# Dockerfile for whatplane application
FROM python:3.8.6-slim-buster

ARG PYTHON_PACKAGES="flask==1.1.2 gunicorn==20.0.4 azure-storage-blob==12.6.0"
ARG APP_DIR="whatplane"
ARG APT_DEPS="curl dumb-init"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
           ${APT_DEPS} \
    && curl -sL https://aka.ms/InstallAzureCLIDeb | bash \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    &&  pip install --no-cache-dir torch==1.7.0+cpu torchvision==0.8.1+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install --no-cache-dir ${PYTHON_PACKAGES}

COPY app ${APP_DIR}/app
COPY src ${APP_DIR}/whatplane

COPY entrypoint.sh /entrypoint.sh
RUN chmod a+x /entrypoint.sh

EXPOSE 5000

WORKDIR "${APP_DIR}/app"

ENTRYPOINT ["/usr/bin/dumb-init", "--", "/entrypoint.sh"]

CMD ["python", "app.py"]