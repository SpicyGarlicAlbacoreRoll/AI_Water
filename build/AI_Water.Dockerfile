FROM osgeo/gdal:ubuntu-full-latest

ENV S3_PYPI_HOST="hyp3-pypi.s3-website-us-east-1.amazonaws.com"

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --fix-missing --no-install-recommends \
    python3.7 \
    python3-pip \
    python3-tk \
    make \
    nano \
    git \
    xauth

RUN pip3 install \
    keras \
    tensorflow \
    pyperclip \
    matplotlib \
    pytest \
    hypothesis \
    pytest-cov \
    mock \
    typing-extensions \
    albumentations \
    tqdm \
    asf_hyp3 \
    --trusted-host "${S3_PYPI_HOST}" --extra-index-url "http://${S3_PYPI_HOST}" hyp3lib



