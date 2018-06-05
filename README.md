# CFCM
This repository contains a publicly available version of
CFCM: segmentation via Coarse to Fine Context Memory,
a paper accepted at MICCAI 2018 for presentation.

In this repository we provide only training/validation code. In the future it will be possibile to use
CFCM to obtain predictions on the Montgomery XRay lung dataset over the cloud
using [TOMAAT](https://tomaatcloud.github.io) and a compatible client
(for example through 3D Slicer and its TOMAAT extension).

The code is provided without guarantees of correctness and functionality.
This implementation is derived directly from the original implementation used for the CFCM paper,
but it has been modified in a way that can result in unexpected behaviors.
Limited support will be given through the means offered by GitHub issues reporting system.

The experiments of the CFCM paper were run on NVIDIA DGX systems with either 16Gb or 32Gb GPUs. 
To the best of our knowledge most experiments can run on workstations with smaller GPUs such as 1080ti, and all the experiments can surely run on GPUs equipped with 24Gb of memory (Quadro P6000 and above).

## Cite

If you use CFCM for your own research please cite our paper using this bibtext:

```
TODO
```

## Local usage
Only recommended for development

### Setup

* install C++ dependencies
  * Ubuntu: `sudo apt install python-opencv ffmpeg`

* (Recommended) create a python 2.7 virtual environment and activate it
```bash 
virtualenv <path-to-virtualenv>
source <path-to-virtualenv>/bin/activate
```
* install the python dependencies
```bash
pip install -r requirements.txt
```

### Usage

* activate the virtualenv 
```bash 
source <path-to-virtualenv>/bin/activate
```
* run training or evaluation
```bash
python cfcm/train.py
```

## Docker usage
This section describes the usage of docker-compose, which allows to avoid verbose shell commands. For a lower-level nvidia-docker experience, please refer to the later section "Usage without docker-compose".

### Prerequisites

* nvidia-container-runtime
* docker-compose >= 1.20

### Setup

#### Register the NVidia docker runtime
By default, docker-compose is not aware of the NVidia docker runtime, so running cfcm could lead to the following error:
```
ERROR: Cannot create container for service cfcm: Unknown runtime specified nvidia   
```

In order to fix it, create a file named /etc/docker/daemon.json with the following content:
```
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```
In order to apply the changes, reload the service configuration and restart it:
```
sudo systemctl deamon-reload
sudo systemctl restart docker
```

#### Login into the NVidia container registry
This image builds on the official NVidia Tensorflow Docker image, which is hosted on the [NVidia container registry](https://www.nvidia.com/en-us/gpu-cloud/deep-learning-containers/). The registry requires authentication in order to download the base image.

* create an account for Nvidia Cloud http://ngc.nvidia.com
* create an API key
* login
```bash
docker login nvcr.io
  username: $oauthtoken 
  password: <YOUR API KEY>
```

In order to avoid having to login manually, you can use [pass](https://www.passwordstore.org/).

### Usage with docker-compose
* update env/local.env with the paths to this folder and to the data folder
* do the same with the paths for your remote setup in env/remote.env

#### Local

```bash
docker-compose run cfcm
```

#### Remote
```bash
docker-compose -f docker-compose.yml -f docker-compose-remote.yml run cfcm
```

### Usage without Docker compose

#### Prerequisites

* nvidia-docker

#### Building
The following command builds a Docker image with all required dependencies, with the name "cfcm":
```bash
cd cfcm
nvidia-docker build -t cfcm .
```

#### Running an interactive shell
This command will start a bash shell in a container:
```bash
nvidia-docker run -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 cfcm bash
```

## Acknowledgments
The code we provide in this repository is a refactoring of the original version. Most of the work for this refactoring has been done by [Marco Esposito](https://github.com/marcoesposito1988).


