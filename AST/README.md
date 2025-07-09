# Audio Spectrogram Transformer

This directory contains code and literature for the Audio Spectrogram Transformer, and training scripts for finetuning it on given datasets


## Directory structure:

- `ast-master` : Repo by the original authors, used as reference for preparation of data for the model
- `finetuned` : Pretrained AST model usage finetuned on the Audioset dataset
- `lit` : Paper by the original authors on AST


## Current Progress:

- Finetuning of the pretrained AST model on ESC-50 dataset was succesful
- Soft Labels for ESC-50 were generated for trasnfer learning
- Working on Preparation of the Balanced Audioset dataset for generation of soft labels is ongoing