// Paths
local root = '/data/users/nilay/the-count/';

// Training
local lr = 2.5e-4;
local decay = 0.00;
local batch_size = 64;
local max_instances = null;
local max_instances_memory = null;
local epochs = 50;
local patience = 5;
local dropout = 0.2;

// Model config
local embed_dim = 768;
local model_path = root + 'saved-experiments/6-layer/';

local cuda_devices = [0, 1];
local cuda_device = 0;

local reader = {
  type: 'cola-reader',
  tokenizer_path: root + 'wordpiece-tokenizer.json',
  max_instances: max_instances,
};

{
  dataset_reader: reader,
  vocabulary: {
    type: 'from_files',
    directory: root + 'data/vocab/',
    padding_token: '[PAD]',
    oov_token: '[UNK]',
  },
  model: {
    type: 'basic_classifier',
    text_field_embedder: {
      type: 'pass_through',
      hidden_dim: 32000,
    },
    seq2vec_encoder: {
      type: 'glue_s2v_encoder',
      model: {
        type: 'from_archive',
        archive_dir: model_path,
      },
    },
    feedforward: {
      type: 'feedforward',
      input_dim: embed_dim,
      hidden_dims: embed_dim * 4,
      activation: 'relu',
      dropout: dropout,
    },
    num_labels: 2,
    dropout: dropout,
  },
  train_data_path: 'train',
  validation_data_path: 'validation',
  test_data_path: 'test',
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
    validation_metric: 'accuracy',
    num_epochs: epochs,
    patience: patience,
    run_sanity_checks: false,
    optimizer: {
      type: 'adam',
      lr: lr,
      weight_decay: decay,
    },
    learning_rate_scheduler: {
      type: 'cosine',
      t_initial: epochs,
    },
    cuda_device: cuda_device,
    grad_norm: 0.25,
    callbacks: [
      {
        type: 'tensorboard',
        should_log_learning_rate: true,
      },
    ],
  },
}
