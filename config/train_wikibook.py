
# WikiBook: full WikiText-2 + BookCorpus training

out_dir = 'out-wikibook'
eval_interval = 200
log_interval = 10
eval_iters = 100
eval_only = False
always_save_checkpoint = False
init_from = 'scratch'

wandb_log = False
wandb_project = 'nanogpt-mqa'
wandb_run_name = 'wikibook'

dataset = 'wikibook'

# Batch and sequence settings
batch_size = 8
block_size = 512
gradient_accumulation_steps = 8

# Model architecture
n_layer = 8
n_head = 12
n_embd = 384
dropout = 0.1
bias = False

# Optimizer
learning_rate = 1e-4
max_iters = 20000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
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
