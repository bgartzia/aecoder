source ENV_AE_DEV/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD
python Scripts/run_training.py -v --config Configs/PS_AE_test.INI
deactivate
