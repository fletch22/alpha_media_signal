#!/usr/bin/env bash

cd /home/jupyter/ || exit

gsutil cp -r gs://api_uploads/twitter/* .

mkdir -p /home/jupyter/alpha_media_signal/
unzip -o project.zip -d /home/jupyter/alpha_media_signal/

cd /home/jupyter || exit

mkdir -p /home/jupyter/overflow_workspace/data/twitter/inference_model_drop
unzip -o twitter_text_label_train.zip -d /home/jupyter/overflow_workspace/data/twitter/inference_model_drop/

mkdir -p /home/jupyter/overflow_workspace/data/financial/quandl/tables/
unzip -o shar_core_fundamentals.zip -d /home/jupyter/overflow_workspace/data/financial/quandl/tables/

unzip -o shar_tickers.zip -d /home/jupyter/overflow_workspace/data/financial/quandl/tables/

mkdir -p /home/jupyter/overflow_workspace/data/financial/quandl/tables/splits_eod
unzip -o eods.zip -d /home/jupyter/overflow_workspace/data/financial/quandl/tables/splits_eod

unzip -o twitter_text_with_proper_labels.zip -d /home/jupyter/overflow_workspace/data/twitter/inference_model_drop

mkdir -p /home/jupyter/overflow_workspace/data/twitter/
cp ./ticker_names_searchable.csv /home/jupyter/overflow_workspace/data/twitter/

mkdir -p /home/jupyter/overflow_workspace/data/twitter/bert/
cp ./reviews.csv /home/jupyter/overflow_workspace/data/twitter/bert/

unzip -o shar_tickers.zip -d /home/jupyter/overflow_workspace/data/financial/quandl/tables/

mkdir -p /home/jupyter/overflow_workspace/data/financial/quandl/tables/shar_actions/
cp ./actions.csv /home/jupyter/overflow_workspace/data/financial/quandl/tables/shar_actions/

mkdir -p /home/jupyter/overflow_workspace/data/cola/cola_public_1.1/cola_public/raw
cp ./in_domain_dev.tsv /home/jupyter/overflow_workspace/data/cola/cola_public_1.1/cola_public/raw/
cp ./in_domain_train.tsv /home/jupyter/overflow_workspace/data/cola/cola_public_1.1/cola_public/raw/
cp ./out_of_domain_dev.tsv /home/jupyter/overflow_workspace/data/cola/cola_public_1.1/cola_public/raw/

cd /home/jupyter/alpha_media_signal || exit
conda install --file ./requirements.txt
pip install -r ./requirements.txt
