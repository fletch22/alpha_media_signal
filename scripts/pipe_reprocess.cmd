
cd ..\

call "activate.bat" alpha_media_signal

python -m python -m ams.marios_workbench.twitter.reprocess_all_tweets.valve

cd .\scripts