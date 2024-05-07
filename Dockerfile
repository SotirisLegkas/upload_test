FROM python:3.12

COPY ./predict_multi_head_tokens.py /predict_multi_head_tokens.py
COPY ./custom_models/multi_head.py /custom_models/multi_head.py
COPY ./requirements.txt /requirements.txt
COPY ./data/final_test/sentences.tsv /final_test/sentences.tsv

WORKDIR /
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
