FROM python:3.8-slim-buster

WORKDIR /app

RUN pip3 install -r requirements.txt
RUN pip3 install predictor==0.1.51.dev0

COPY app.py .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=8085"]