// Paths
local root = '/home/offendo/src/capstone/';

// Training
local context = 50;
local max_instances = 20;
local lr = 0.001;
local batch_size = 16;
local epochs = 50;
local patience = 10;

// Model config
local num_hidden_layers = 4;
local embedding_dim = 64;
local intermediate_dim = 128;
local num_attention_heads = 4;

{
  dataset_reader: {
    type: 'wikitext-reader',
    context: context,
    tokenizer: {
      type: 'wikitext-tokenizer',
      tokenizer_path: root + 'unigram-tokenizer.json',
      add_special_tokens: true,
    },
    token_indexers: {
      tokens: {
        type: 'single_id',
        namespace: 'tokens',
      },
    },
    max_instances: 20,
    manual_multiprocess_sharding: true,
    manual_distributed_sharding: true,
  },
  vocabulary: {
    type: 'from_files',
    directory: root + 'data/vocab/',
    padding_token: '[PAD]',
    oov_token: '[UNK]',
  },
  model: {
    type: 'simple-transformer-language-model',
    num_hidden_layers: num_hidden_layers,
    hidden_size: embedding_dim,
    intermediate_size: intermediate_dim,
    num_attention_heads: num_attention_heads,
    embedder: {
      type: 'basic',
      token_embedders: {
        tokens: {
          type: 'embedding',
          embedding_dim: embedding_dim,
        },
      },
    },
  },
  train_data_path: root + 'data/wikitext-103-raw/wiki.train.raw',
  validation_data_path: root + 'data/wikitext-103-raw/wiki.valid.raw',
  data_loader: {
    type: 'multiprocess',
    batch_size: batch_size,
    shuffle: true,
    num_workers: 4,
    start_method: 'spawn',
  },
  trainer: {
    type: 'gradient_descent',
    validation_metric: '-perplexity',
    num_epochs: epochs,
    patience: patience,
    optimizer: {
      type: 'adam',
      lr: lr,
    },
    cuda_device: -1,
  },
}
