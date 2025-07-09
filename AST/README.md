# Audio Spectrogram Transformer

This directory contains code and literature for the Audio Spectrogram Transformer, and training scripts for finetuning it on given datasets


## Directory structure:

- `ast-master` : Repo by the original authors, used as reference for preparation of data for the model
- `finetuned` : Pretrained AST model usage finetuned on the Audioset dataset
- `lit` : Paper by the original authors on AST


## Finetuned directory structure:

Currenlty the main implementation being used is the finetuned model

- `exp`: contains the `esc` directory for saved models, and the `landing` directory for the evaluated soft labels
- `combined_utils.py`: Contains code for varied use, Specialized Preprocessing scripts, Evalutaion metric, Training scipts, Custom FBank datasets, Generation scipts for Model Distillation, etc
- `ESC_run.ipynb`: Contains the entire run over ESC dataset, using the new `ASetFineAnyAST` model
- `data.zip` contains the esc dataset used in the above run




## Current Progress:

- Finetuning of the pretrained AST model on ESC-50 dataset was succesful
- Soft Labels for ESC-50 were generated for trasnfer learning

-----------------------------------

- Working on Preparation of the Balanced Audioset dataset
- Current Issue: Working Audioset dataset has inconsistent "corrupted" files which are obstructing preprocessing attempts