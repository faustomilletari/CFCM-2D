FROM nvcr.io/nvidia/tensorflow:18.05-py2

RUN apt-get update

RUN apt-get install -y python-opencv
RUN apt-get install -y ffmpeg

COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN rm requirements.txt

COPY . /opt/app
RUN python setup.py install
