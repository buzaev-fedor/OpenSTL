method = 'simvp_adr'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 64
hid_T = 512
N_T = 8
N_S = 4
mlp_ratio = 8.
drop = 0.


# ADR specific parameters
adr_layers = 2
adr_hid_dim = 64
device = 'cuda'  # Add device parameter

# training
lr = 1e-3
batch_size = 200
# drop_path = 0.1
sched = 'onecycle'

# data
dataset_name = 'mmnist'
in_shape = [10, 1, 64, 64]
pre_seq_length = 10
aft_seq_length = 10

# log and save
log_step = 5
save_every = 100
val_every = 20
epoch = 2000