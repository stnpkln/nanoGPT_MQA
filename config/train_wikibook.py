
# WikiBook: 12-head MQA experiment

out_dir = 'out-wikibook-12head'
wandb_log = False
wandb_project = 'nanogpt-mqa'
wandb_run_name = 'wikibook-12head'

dataset = 'wikibook'
init_from = 'scratch'

# Batch settings
batch_size = 8
block_size = 512
gradient_accumulation_steps = 8

# Model architecture
n_layer = 8
n_head = 12
n_embd = 384
dropout = 0.1

# Training duration
max_iters = 20000
lr_decay_iters = 20000

# Evaluation
eval_interval = 500
eval_iters = 100
log_interval = 10

# Optimizer
learning_rate = 3e-4
min_lr = 3e-5
warmup_iters = 500
weight_decay = 1e-1

compile = False
