// Paths
// local root = '/data/users/nilay/the-count/';
local root = '/data/users/malsalih/Model_Distillation/';

// Training
local context = 96;
local lr = 0.0005;
local batch_size = 32;
local max_instances = 1024;
local max_instances_memory = 1024;
local epochs = 50;
local patience = 10;
local dropout = 0.3;

// Model config
local num_layers = 4;
local embedding_dim = 128;
local max_positions = 256;
local hidden_dim = 196;
local num_attention_heads = 4;
local activation = 'relu';

local cuda_devices = [0, 3];

local reader = {
  type: 'wikitext-reader',
  context: context,
  tokenizer: {
    type: 'wikitext-tokenizer',
    tokenizer_path: root + 'wordpiece-tokenizer.json',
    add_special_tokens: true,
  },
  token_indexers: {
    tokens: {
      type: 'single_id',
      namespace: 'tokens',
    },
  },
  exclusive: true,
  split_on: 'sentence',
  min_context_len: 5,
  max_instances: max_instances,
  manual_multiprocess_sharding: true,
  manual_distributed_sharding: true,
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
    type: 'simple-transformer-language-model',
    embedder: {
      type: 'basic',
      token_embedders: {
        tokens: {
          type: 'embedding',
          embedding_dim: 128,
        },
      },
    },
    decoder: {
      type: 'gpt2-transformer-decoder',
      input_dim: 128,
      hidden_dim: hidden_dim,
      num_attention_heads: num_attention_heads,
      num_layers: num_layers,
      activation: activation,
      dropout: dropout,
    },
    embedding_dim: embedding_dim,
    max_positions: max_positions,
  },
  train_data_path: root + 'data/wikitext-103-raw/wiki.train.raw',
  validation_data_path: root + 'data/wikitext-103-raw/wiki.valid.raw',
  test_data_path: root + 'data/wikitext-103-raw/wiki.test.raw',
  data_loader: {
    type: 'multiprocess',
    batch_size: batch_size,
    shuffle: true,
    max_instances_in_memory: max_instances_memory,
    num_workers: 4,
    start_method: 'fork',
  },
  validation_data_loader: {
    type: 'multiprocess',
    batch_size: batch_size,
    shuffle: false,
    max_instances_in_memory: max_instances_memory,
    num_workers: 4,
    start_method: 'fork',
  },
  trainer: {
    type: 'gradient_descent',
    validation_metric: '-perplexity',
    num_epochs: epochs,
    patience: patience,
    optimizer: {
      type: 'adamw',
      lr: lr,
      weight_decay: 0.1,
    },
  },
  distributed: {
    cuda_devices: cuda_devices,
  },
}
