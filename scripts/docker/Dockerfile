ARG TF_VERSION=latest
FROM tensorflow/tensorflow:${TF_VERSION}-gpu

RUN apt update
RUN apt install -y python3-dev libgdal-dev
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade setuptools requests numpy six
RUN python3 -m pip install GDAL==`gdal-config --version` --global-option=build_ext --global-option="-I/usr/include/gdal/"

COPY ./setup.py /src/
RUN cd /src/ && python3 setup.py egg_info && sed -i '/tensorflow/d' delta.egg-info/requires.txt && python3 -m pip install -r delta.egg-info/requires.txt
# this might only work with latest and 2.1
RUN if dpkg --compare-versions "$TF_VERSION" "lt" "2.3"; then \
       python3 -m pip install tensorflow_addons==0.9.1; \
    elif dpkg --compare-versions "$TF_VERSION" "lt" "2.4"; then \
       python3 -m pip install tensorflow_addons==0.13.0; \
    else \
       python3 -m pip install tensorflow_addons; \
    fi
COPY ./ /src/delta/
RUN cd /src/delta && python3 -m pip install --no-dependencies .
