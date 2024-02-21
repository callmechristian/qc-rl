FROM ubuntu:20.04, focal, focal-20221019
ADD file:7633003155a1059419aa1a6756fafb6e4f419d65bff7feb7c945de1e29dccb1e in /
CMD ["bash"]
ENV DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y curl
ENV LANG=C.UTF-8
apt-get update && apt-get install -y python3.9
py -m pip --no-cache-dir install --upgrade "pip<20.3" setuptools
ln -s $(which python3) /usr/local/bin/python
ARG TF_PACKAGE=tensorflow
ARG TF_PACKAGE_VERSION=
|2 TF_PACKAGE=tensorflow-cpu TF_PACKAGE_VERSION=2.11.0 /bin/sh -c python3 -m pip install --no-cache-dir ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}
COPY file:946fcda4ffd2c64c58e8e7354df88b38d2fd1032e3c765068bf9345ce8d25fb7 in /etc/bash.bashrc
|2 TF_PACKAGE=tensorflow-cpu TF_PACKAGE_VERSION=2.11.0 /bin/sh -c chmod a+rwx /etc/bash.bashrc
