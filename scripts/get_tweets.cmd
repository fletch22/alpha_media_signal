cd ..\ || exit

SET PYTHONPATH=%CD%

call "activate.bat" alpha_media_signal

python -m ams.services.twitter_service

cd .\scripts || exit

conda deactivate