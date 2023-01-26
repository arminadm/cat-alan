FROM python:3.9-slim-buster

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . /app/

RUN apt-get update && apt-get upgrade -y && \
    apt-get install libsndfile-dev -y
RUN chmod +x firstTimeSetup.sh && \
        ./firstTimeSetup.sh

