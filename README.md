# Zoionent

This Project is an extenion of the of submission (under the same name) to the [IEEE Comsoc Student competiion 2024](https://www.comsoc.org/membership/ieee-comsoc-student-competition/winners), where the submission earned an honorary mention (ie top 15 worldwide).

This repo aims to provide a more powerful and refined approach over the original submission on the same problem, with use of other architecutres like AST, and CRNN and more advance edge devices.

## Files and Folders

- `crnn`: contains the literature and the subsequent implementation of the CRNN model
- `AST`: contains the literature and implementation of the AST model
- `legacy_files`: Contains methodology abstract and experimentation submitted to IEEE, based on the previous implementation
- `melspec.ipynb`: Contains code for loading audio files and converting them to mel log spectrograms

## Prerequisites

The required libaries can be installed using the requirements file
NOTE: These will be brought down once experimentation is reached

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

- Models for AST and CRNN ready
- Preprocessing scripts ready
- Soft Labels for ESC-50 dataset ready for benchmarking

---------------------------------------------------------

- Preparation for the Balanced Audioset dataset pending
- Current limitations is a reliable source for the dataset along with hardware resources to process the entire dataset
