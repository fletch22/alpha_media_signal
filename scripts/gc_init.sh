#!/usr/bin/env bash

cd /home/jupyter/ || exit

gsutil cp -r gs://api_uploads/twitter/gc_install.sh .

chmod +x ./gc_install.sh

chown jupyter ./gc_install.sh





