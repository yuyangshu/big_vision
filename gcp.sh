# cd ~/playground/tensorflow_datasets
gcloud storage cp --recursive --no-clobber imagenet_v2 gs://imagenet-1k
gcloud storage cp --recursive --no-clobber imagenet2012_real gs://imagenet-1k
gcloud storage cp --recursive --no-clobber imagenet2012 gs://imagenet-1k


# https://cloud.google.com/tpu/docs/storage-buckets
# this principal is not visible from the console; after creation need to grant bucket permission
gcloud beta services identity create --service tpu.googleapis.com --project vit-test-ys
# Service identity created: service-969988812741@cloud-tpu.iam.gserviceaccount.com


# README.md
# https://cloud.google.com/tpu/docs/run-calculation-jax
# https://cloud.google.com/tpu/docs/preemptible#console

# cd ~/playground
export NAME=default-vm
# export ZONE=us-central2-b
export ZONE=europe-west4-a
export DATA_BUCKET=imagenet-1k
export OUTPUT_BUCKET=workdir-ys

# create TPU VM
# gcloud compute tpus tpu-vm create $NAME --zone=$ZONE --accelerator-type=v4-8 --version=tpu-vm-v4-base --preemptible
gcloud compute tpus tpu-vm create $NAME --zone=$ZONE --accelerator-type=v3-8 --version=tpu-vm-base --preemptible

# check created/preempted/deleted
gcloud compute tpus tpu-vm list --zone=$ZONE
# NAME        ZONE           ACCELERATOR_TYPE  TYPE  TOPOLOGY  NETWORK  RANGE          STATUS
# default-vm  us-central1-a  v3-8              V3    2x2       default  10.128.0.0/20  READY

# copy repo
gcloud compute tpus tpu-vm scp --recurse big_vision/big_vision $NAME: --zone=$ZONE --worker=all

# train
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command "TFDS_DATA_DIR=gs://$DATA_BUCKET bash big_vision/run_tpu.sh big_vision.train --config big_vision/configs/vit_s16_i1k.py --workdir gs://$OUTPUT_BUCKET/`date '+%m-%d_%H%M'`"

# clean up
gcloud compute tpus tpu-vm delete $NAME --zone=$ZONE
# NAME        ZONE            ACCELERATOR_TYPE  TYPE  TOPOLOGY  NETWORK  RANGE          STATUS
# default-vm  europe-west4-a  v3-32             V3    4x4       default  10.164.0.0/20  DELETING
