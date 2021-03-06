// Paths
local root = '/data/users/aukking/Model_Distillation/';

// Training
local sequence_length = 256;
local lr = 1e-4;
local decay = 1e-4;
local batch_size = 8;
local max_instances = null;
local max_instances_memory = null;
local epochs = 50;
local patience = 5;
local dropout = 0.3;

// Student
local num_layers = 10;
local embedding_dim = 768;
local hidden_dim = embedding_dim * 4;
local num_attention_heads = 12;

// Model config
local forward_path = root + '/saved-experiments/138M-model/';
local backward_path = '/data/users/aukking/Model_Distillation/saved-experiments/backwards-baseline-138M-4/inter_results/model.tar.gz';

// Hyper params
local temperature = 3;
local hard_label_weight = 0.3;

local cuda_devices = [1, 2];
local cuda_device = 1;

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
    type: 'teacher-student-language-model',
    temperature: temperature,
    hard_label_weight: hard_label_weight,
    student: {
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
    teacher: {
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
      type: 'cosine',
      t_initial: epochs,
    },
    cuda_device: cuda_device,
    grad_norm: 0.25,
    callbacks: [
      {
        type: 'tensorboard',
      },
    ],
  },
  // distributed: {
  //   cuda_devices: cuda_devices,
  // },
}
