# Zoionent (Extenion WIP)

This Project is an extenion of the of submission (under the same name) to the [IEEE Comsoc Student competiion 2024](https://www.comsoc.org/membership/ieee-comsoc-student-competition/winners), where the submission earned an honorary mention (ie top 15 worldwide).

This repo aims to provide a more powerful and refined approach over the original submission on the same problem, with use of other architecutres like AST, and CRNN and more advance edge devices.

## Files and Folders

- `crnn`: contains the literature and the subsequent implementation of the CRNN model
- `AST`: contains the literature and implementation of the AST model
- `preprocessing` (to be pushed): contains scripts for loading audio files and converting them to mel log spectrograms
- `datasets` (to be pushed): contains datasets for pretraining and training for the target problem
- `detection` (to be created): contains the low power/less accurate/passive detection model weights and scripts.
- `classification` (to be created): contains the high power/more accurate/active classifcation model weights and scripts.
- `old_abstract.pdf`: The orginal abstract which was used in submission in 2024
- `ZoionNET.pdf` (in progress): Current paper descibing the methodology and experimentation

## Prerequisites (to be brought down)

The required libaries can be installed using the requirements file

Currenlty the Prerequisites include:

- Numpy
- scipy
- ipython
- Pytorch
- Torchinfo
- Torchvision
- Pandas
- PIL
- Librosa


```bash
pip install -r requirements.txt
```

### Current Progress/ To be done

- model for crnn ready
- model for ast to be reworked
- preprocessing scripts ready (to be pushed)
- old abstract to be pushed
- work on script for integrating low + high power models started
