
cd ..\

call "activate.bat" alpha_media_signal

python -m ams.marios_workbench.twitter.import_and_predict.valve

cd .\scripts