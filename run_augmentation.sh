source ENV_AE_DEV/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD
python Scripts/run_preprocessing.py --config Configs/PS_AE_test.INI -v
deactivate


