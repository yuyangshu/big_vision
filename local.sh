# cd ~/playground/big_vision
export TFDS_DATA_DIR=~/playground/tensorflow_datasets
source ../venv/bin/activate

XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -m big_vision.train --config big_vision/configs/vit_s16_i1k.py --workdir ../workdirs/`date '+%m-%d_%H%M'`
