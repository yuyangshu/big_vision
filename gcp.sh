# export PROJECT_ID=vit-test-ys
export PROJECT_ID=test

# configure gcloud cli
gcloud config set account $EMAIL
gcloud config set project $PROJECT_ID

# enable TPU
gcloud services enable tpu.googleapis.com

# create Cloud TPU Service Account for project
# https://cloud.google.com/tpu/docs/storage-buckets
# this principal is not visible from the console; after creation need to grant bucket permission
gcloud beta services identity create --service tpu.googleapis.com --project $PROJECT_ID
# Service identity created: service-969988812741@cloud-tpu.iam.gserviceaccount.com
# Service identity created: service-695580597349@cloud-tpu.iam.gserviceaccount.com


# copy training data
export BUCKET_NAME=vit-test-ys

# create the bucket
# then add "Storage Legacy Bucket Owner" permission from the bucket to the service identity from the console

# cd ~/playground
gcloud storage cp --recursive --no-clobber tensorflow_datasets/imagenet_v2 gs://$BUCKET_NAME/tensorflow_datasets
gcloud storage cp --recursive --no-clobber tensorflow_datasets/imagenet2012_real gs://$BUCKET_NAME/tensorflow_datasets
gcloud storage cp --recursive --no-clobber tensorflow_datasets/imagenet2012 gs://$BUCKET_NAME/tensorflow_datasets


# README.md
# https://cloud.google.com/tpu/docs/run-calculation-jax
# https://cloud.google.com/tpu/docs/preemptible#console

# cd ~/playground
export BUCKET_NAME=vit-test-ys
export VM_NAME=default-vm
# export ZONE=us-central2-b
export ZONE=europe-west4-a

# create TPU VM
# gcloud compute tpus tpu-vm create $VM_NAME --zone=$ZONE --accelerator-type=v4-8 --version=tpu-vm-v4-base --preemptible
gcloud compute tpus tpu-vm create $VM_NAME --zone=$ZONE --accelerator-type=v3-8 --version=tpu-vm-base --preemptible

# check created/preempted/deleted
gcloud compute tpus tpu-vm list --zone=$ZONE
# NAME        ZONE           ACCELERATOR_TYPE  TYPE  TOPOLOGY  NETWORK  RANGE          STATUS
# default-vm  us-central1-a  v3-8              V3    2x2       default  10.128.0.0/20  READY

# copy repo
gcloud compute tpus tpu-vm scp --recurse big_vision/big_vision $VM_NAME: --zone=$ZONE --worker=all

# train
gcloud compute tpus tpu-vm ssh $VM_NAME --zone=$ZONE --worker=all --command "TFDS_DATA_DIR=gs://$BUCKET_NAME/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train --config big_vision/configs/vit_s16_i1k.py --workdir gs://$BUCKET_NAME/workdirs/`date '+%m-%d_%H%M'`"

# clean up
gcloud compute tpus tpu-vm delete $VM_NAME --zone=$ZONE
# NAME        ZONE            ACCELERATOR_TYPE  TYPE  TOPOLOGY  NETWORK  RANGE          STATUS
# default-vm  europe-west4-a  v3-32             V3    4x4       default  10.164.0.0/20  DELETING
