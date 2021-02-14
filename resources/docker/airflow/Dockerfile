#FROM apache/airflow:v1-10-stable-python3.8-build
FROM puckel/docker-airflow:1.10.9

RUN pip install --user psycopg2-binary cryptography==3.3.1

ENV AIRFLOW_HOME=/usr/local/airflow

COPY ./airflow.cfg /usr/local/airflow/airflow.cfg

WORKDIR /

COPY ./requirements.txt ./

USER airflow

ENV PATH="/usr/local/airflow/.local/bin:${PATH}"

RUN /usr/local/bin/python -m pip install --upgrade pip && pip install --user -r requirements.txt