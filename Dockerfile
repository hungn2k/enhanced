FROM python:3.7

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/code/shadow_remove

RUN mkdir /code
WORKDIR /code

COPY ./requirements.txt .
RUN pip install -r requirements.txt && rm -f requirements.txt
COPY ./shadow_remove /code/shadow_remove


