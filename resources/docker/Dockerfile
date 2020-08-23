FROM jupyter/pyspark-notebook:54462805efcb

USER root

RUN apt-get update -y && apt-get install -y zip

COPY ./tmp/pip_requirements.txt ./pip_requirements.txt

USER jovyan

RUN pip install -r ./pip_requirements.txt

#RUN conda install --file /requirements.txt
