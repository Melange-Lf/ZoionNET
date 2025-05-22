import sys
import os
sys.path.append(os.path.dirname(__file__))
import torch
from combined import FbankDataset, train_model, run_inference
from models.ast_models import ASTModel

# Configuration dictionary
config_dict = {
    'num_mel_bins': 128,
    'target_length': 512, # {'audioset':1024, 'esc50':512, 'speechcommands':128}
    'loss' : 'CE',
    'mode':'train', 
    'mean':-6.6268077, # ESC -6.6268077
    'std' : 5.358466, # ESC 5.358466
    'fstride' : 10,
    'tstride' : 10,
    'input_fdim' : 128,
    'input_tdim' : 512,
    'imagenet_pretrain' : True,
    'audioset_pretrain' : True,
    'model_size' : 'base384',
    'epochs' : 25,
    'lr' : 1e-5, # audioset pretrain is false, then one order up
    'weight_decay' : 5e-7,
    'betas' : (0.95, 0.999),
    'lrscheduler_start' : 5,
    'lrscheduler_step' : 1,
    'lrscheduler_decay' : 0.85,
    'print_freq' : 0,
    'exp_dir' : "AST/ast-master/egs/esc50/exp/custom_run"
}

# Paths
train_json = "AST/ast-master/egs/esc50/data/datafiles_fbank/esc_train_data_1.json"
eval_json = "AST/ast-master/egs/esc50/data/datafiles_fbank/esc_eval_data_1.json"
label_csv = "AST/ast-master/egs/esc50/data/esc_class_labels_indices.csv"
train_out_csv = "AST/ast-master/egs/esc50/exp/custom_run/esc_train_1.csv"
eval_out_csv = "AST/ast-master/egs/esc50/exp/custom_run/esc_eval_1.csv"

# Dataloaders
train_dataset = FbankDataset(train_json, label_csv=label_csv)
eval_dataset = FbankDataset(eval_json, label_csv=label_csv)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True
)
eval_loader = torch.utils.data.DataLoader(
    eval_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True
)

# Model
model = ASTModel(
    label_dim=len(train_dataset.index_dict),
    fstride=config_dict['fstride'],
    tstride=config_dict['tstride'],
    input_fdim=config_dict['input_fdim'],
    input_tdim=config_dict['input_tdim'],
    imagenet_pretrain=config_dict['imagenet_pretrain'],
    audioset_pretrain=config_dict['audioset_pretrain'],
    model_size=config_dict['model_size']
)

# Convert model to float32
model = model.float()

# Train
os.makedirs(config_dict['exp_dir'], exist_ok=True)
train_model(model, train_loader, eval_loader, config_dict)

# Inference on train and eval sets
run_inference(model, train_json, label_csv, train_out_csv)
run_inference(model, eval_json, label_csv, eval_out_csv) 