// Paths
local root = '/data/users/nilay/the-count/';

// Training
local sequence_length = 256;
local lr = 2.5e-4;
local decay = 0.00;
local batch_size = 32;
local max_instances = null;
local max_instances_memory = null;
local epochs = 30;
local patience = 5;
local dropout = 0.1;

// Model config
local num_layers = 16;
local embedding_dim = 768;
local hidden_dim = embedding_dim * 4;
local num_attention_heads = 12;
local activation = 'relu';

local cuda_devices = [1, 2];
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
    type: 'simple-transformer-language-model',
    embedding_dim: embedding_dim,
    embedder: {
      embedding_dim: embedding_dim,
    },
    pos_embedder: {
      embedding_dim: embedding_dim,
      num_embeddings: sequence_length,
    },
    decoder: {
      type: 'gpt2-transformer-decoder',
      input_dim: embedding_dim,
      hidden_dim: hidden_dim,
      num_attention_heads: num_attention_heads,
      num_layers: num_layers,
      dropout: dropout,
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
    // learning_rate_scheduler: {
    //   type: 'cosine_with_warmup',
    //   num_training_steps: 14085 * epochs,
    //   num_warmup_steps: 5000,
    // },
    cuda_device: cuda_device,
    grad_norm: 0.25,
    callbacks: [
      {
        type: 'tensorboard',
        should_log_learning_rate: true,
      },
    ],
  },
  // distributed: {
  //   cuda_devices: cuda_devices,
  // },
}
