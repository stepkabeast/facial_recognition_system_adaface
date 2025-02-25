ARG REPOSITORY="docker.io"
FROM python:3.8-bookworm

RUN rm -f /var/lib/dpkg/statoverride && rm -f /var/lib/dpkg/lock && dpkg --configure -a
RUN apt-get update && apt-get install -y -o Dpkg::Options::="--force-confold" ffmpeg libsm6 libxext6

ENV PYTHONPATH "${PYTHONPATH}:/face_index_root"

ADD requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

RUN mkdir /face_index_root
ADD face_index /face_index_root/face_index
ADD pg_model/ /face_index_root/pg_model
ADD VERSION /face_index_root


CMD ["python3.8", "/face_index_root/face_index/main.py"]