// Paths
local root = '/data/users/aukking/Model_Distillation/';

// Training
local sequence_length = 256;
local lr = 2.5e-4;
local decay = 0.00;
local batch_size = 16;
local max_instances = null;
local max_instances_memory = null;
local epochs = 50;
local cosine_epochs = 49;
local patience = 3;
local dropout = 0.1;

// Model config
local forward_path = '/data/users/nilay/the-count/saved-experiments/138M-model/';
local backward_path = root + '/saved-experiments/backwards-baseline-138M-4/inter_results/';

local cuda_devices = [2, 3];
local cuda_device = 0;

local train_reader = {
  type: 'wikitext-reader',
  sequence_length: sequence_length,
  tokenizer_path: root + 'wordpiece-tokenizer.json',
  max_instances: max_instances,
};

local eval_reader = {
  type: 'wikitext-reader',
  sequence_length: sequence_length,
  tokenizer_path: root + 'wordpiece-tokenizer.json',
  max_instances: max_instances,
};

{
  dataset_reader: train_reader,
  validation_dataset_reader: eval_reader,
  vocabulary: {
    type: 'from_files',
    directory: root + 'data/vocab/',
    padding_token: '[PAD]',
    oov_token: '[UNK]',
  },
  model: {
    type: 'dual-directional-language-model',
    forward_model: {
        type: 'from_archive',
        archive_file: '/data/users/nilay/the-count/saved-experiments/138M-model/',
    },
    backward_model: {
        type: 'from_archive',
        archive_file: backward_path,
    },
  },
  train_data_path: root + 'data/wikitext-103/wiki.train.tokens',
  validation_data_path: root + 'data/wikitext-103/wiki.valid.tokens',
  test_data_path: root + 'data/wikitext-103/wiki.test.tokens',
  data_loader: {
    type: 'multiprocess',
    batch_size: batch_size,
    shuffle: true,
    max_instances_in_memory: max_instances_memory,
    num_workers: 0,
    start_method: 'fork',
  },
  validation_data_loader: {
    type: 'multiprocess',
    batch_size: batch_size,
    shuffle: false,
    max_instances_in_memory: max_instances_memory,
    num_workers: 0,
    start_method: 'fork',
  },
  trainer: {
    type: 'gradient_descent',
    validation_metric: '-perplexity',
    num_epochs: epochs,
    patience: patience,
    run_sanity_checks: false,
    optimizer: {
      type: 'adam',
      lr: lr,
      weight_decay: decay,
    },
    learning_rate_scheduler: {
      type: 'combined',
      schedulers: [
      [1, {
        type: 'linear_with_warmup',
        warmup_steps: 10000,
        num_epochs: 1,
      }],
      [epochs - 1, {
        type: 'cosine',
        t_initial: epochs-1,
      }],
      ],
    },
    cuda_device: cuda_device,
    grad_norm: 0.25,
    callbacks: [
      {
        type: 'tensorboard',
      },
    ],
  },
/*  distributed: {
    cuda_devices: cuda_devices,
  },*/
}
