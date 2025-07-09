
# CRNN Section

This subsection contains the literature used, and the subsequent implementation of the CRNN architecture.

This model is to be used in the detection part of the problem, and is the student model for the AST model.

## Files

The subsection contains the following files:

- `lit` directory: contains the pdf files used as source for the crnn implementation.
- `base 1 ref3` pdf: Used as the main paper to translate its ideas into code, the two other papeers, (base1 and base1 mod1) were used as further references.
- `code` directory: Contains `crnn.ipynb`, the main implementation, `engine.py` and `utils.py`
` `utils.py`: Contains other utility functions for saving/loading models, loading data, base script sourced from [mrdbourke](https://github.com/mrdbourke) for simplicity, then modfied to suit our specific purposes.
- `engine.py`: Contains functions for training purpose of pytorch models (credit: [mrdbourke](https://github.com/mrdbourke)).
- `run.ipynb`: The original run over ESC-50 without pretraining on a larger dataset to test robustness of the model

## Architecture Implementation(WIP)

Note the orginal archtecture is specialized for SED (sound event detection) in which it detects presence of every sound class for every single time step, rather then just a single detection output for an audio file.
The model has been modifed somewhat to aggregate the inferences of each time step into a single ouput.

To do this, the hidden states for each time step (each orignally inteded to be fed into final linear layer for inference), are fed into an encoder layer (this layer is copied over for every time step) to reduce the number of hidden states.

The reduced hidden states are then concatatenated together to be fed through a MLP to get final inference.


### Progress (TODO)

- Custom Pytorch Dataset for loading numpy files, and their respective softlabels to be implemented from csv...
- Convert the `crnn.ipynb` to a .py file to make it easier to be sourced in other files...
- Add to the README, especially the Architecture Section
- Add Hyperparameter section





