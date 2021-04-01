cd ..\ || exit

SET PYTHONPATH=%CD%

call "activate.bat" alpha_media_signal

python -m ams.marios_workbench.twitter.master.valve

cd .\scripts || exit

conda deactivate