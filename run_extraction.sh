source ENV_AE_DEV/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD
python Scripts/run_extraction.py -v --config Configs/surfInspAE.INI
deactivate
