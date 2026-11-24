out_dir = 'out-wikitext2-mha'
eval_interval = 200
log_interval = 10
eval_iters = 100
eval_only = False
always_save_checkpoint = False
init_from = 'scratch'
early_stopping_patience = 6

wandb_log = False
wandb_project = 'nanogpt-mqa'
wandb_run_name = 'wikitext2-mha'

dataset = 'wikitext2'

# Batch and sequence settings
batch_size = 32
block_size = 256
gradient_accumulation_steps = 1

# Model architecture
n_layer = 4
n_head = 4
n_embd = 384
dropout = 0.2
bias = False
mqa = False

# Optimizer
learning_rate = 3e-4
max_iters = 20000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# Learning-rate schedule
decay_lr = True
warmup_iters = 500
lr_decay_iters = 20000
min_lr = 3e-5

# Hardware
device = 'cuda'
dtype = 'float16'
compile = False