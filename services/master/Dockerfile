FROM python:3.8-slim-buster

ARG MODEL_DIR

WORKDIR /app

RUN pip3 install -r requirements.txt

COPY ["app.py","./"]

CMD ["python3", "./app.py"]
