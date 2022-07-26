# Copyright 2022 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=line-too-long
r"""A config for training MLP-Mixer-B/16 model on ILSVRC-2012 ("ImageNet-1k").

Achieves 76.3% top-1 accuracy on the test split in 2h11m on TPU v3-128
with 300 epochs. A shorter 60 epochs run is expected to get to 70.5% in 27m.

big_vision.train \
    --config big_vision/configs/mlp_mixer_i1k.py \
    --workdir gs://[your_bucket]/big_vision/`date '+%m-%d_%H%M'` \
"""

from big_vision.configs.common_fewshot import get_fewshot_lsr
import ml_collections as mlc


def get_config(mode=None):
  """Config for training Mixer on i1k."""
  config = mlc.ConfigDict()

  config.dataset = 'imagenet2012'
  config.train_split = 'train[:99%]'
  config.cache_raw = True  # Needs up to 120GB of RAM!
  config.num_classes = 1000
  config.init_head_bias = -6.9
  config.loss = 'sigmoid_xent'

  config.pp_train = (
      'decode_jpeg_and_inception_crop(224)'
      '|flip_lr'
      '|randaug(2,15)'
      '|value_range(-1, 1)'
      '|onehot(1000, key="label", key_result="labels")'
      '|keep("image", "labels")'
  )
  ppv = (
      'decode'
      '|resize_small(256)|central_crop(224)'
      '|value_range(-1, 1)'
      '|onehot(1000, key="{lbl}", key_result="labels")'
      '|keep("image", "labels")'
  )

  config.batch_size = 4096
  config.total_epochs = 300

  config.shuffle_buffer_size = 250_000  # Per host, so small-ish is ok.

  config.log_training_steps = 50
  config.ckpt_steps = 1000
  config.ckpt_timeout = 1

  config.prefetch_to_device = 2

  # Model section
  config.model_name = 'mlp_mixer'
  config.model = dict()
  config.model.variant = 'B/16'
  config.model.stoch_depth = 0.1

  config.mixup = dict(fold_in=None, p=0.5)

  # Optimizer section
  config.optax_name = 'scale_by_adam'
  config.grad_clip_norm = 1.

  config.lr = 0.001
  config.wd = 1e-4
  config.schedule = dict(
      decay_type='linear',
      warmup_steps=10_000,
      linear_end=1e-5,
  )

  # Eval section
  eval_common = dict(
      type='classification',
      dataset='imagenet2012',
      pp_fn=ppv.format(lbl='label'),
      loss_name=config.loss,
      log_steps=2500,  # Very fast O(seconds) so it's fine to run it often.
  )
  config.evals = {}
  config.evals.train = {**eval_common, 'split': 'train[:2%]'}
  config.evals.minival = {**eval_common, 'split': 'train[99%:]'}
  config.evals.val = {**eval_common, 'split': 'validation'}
  config.evals.v2 = {**eval_common, 'dataset': 'imagenet_v2', 'split': 'test'}

  config.evals.real = dict(**eval_common)
  config.evals.real.dataset = 'imagenet2012_real'
  config.evals.real.split = 'validation'
  config.evals.real.pp_fn = ppv.format(lbl='real_label')

  config.fewshot = get_fewshot_lsr()

  if mode == 'gpu8':
    config.total_epochs = 60
    config.batch_size = 512
    config.cache_raw = False
  if mode == 'regression_test':
    config.total_epochs = 60

  return config
