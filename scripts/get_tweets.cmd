SET PYTHONPATH="C:\Users\Chris\workspaces\alpha_media_signal"

cd ..\ || exit

REM call "activate.bat" alpha_media_signal

python -m ams.services.twitter_service

cd .\scripts || exit

REM call "conda deactivate"