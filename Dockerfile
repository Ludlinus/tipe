FROM python:latest 

WORKDIR /usr/local/bin

COPY requirements.txt
COPY run_NN.py
COPY config_1.txt

RUN pip install -r requirements.txt

CMD ["run_NN.py"]
