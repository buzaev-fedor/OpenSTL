method = 'simvp_adr'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gsta'  # gsta
hid_S = 128
hid_T = 1024
N_T = 24
N_S = 4
mlp_ratio = 8.
drop = 0.
drop_path = 0.1

# ADR specific parameters
adr_layers = 2
adr_hid_dim = 32
device = 'cuda'  # Add device parameter

# training
lr = 1e-3
batch_size = 56
drop_path = 0.1
sched = 'onecycle'

# data
dataset_name = 'mmnist'
in_shape = [10, 1, 64, 64]
pre_seq_length = 10
aft_seq_length = 10

# log and save
log_step = 5
save_every = 100
val_every = 10
epoch = 1000