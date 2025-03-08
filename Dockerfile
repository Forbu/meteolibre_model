### torch image
FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*
### install python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
### copy the code
COPY . /app
WORKDIR /app
### set the entrypoint
ENTRYPOINT ["python", "train.py"]