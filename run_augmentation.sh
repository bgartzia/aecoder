source ENV_AE_DEV/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD
python Scripts/run_preprocessing.py --config Configs/surfMeanCurvAE.INI -v
deactivate


