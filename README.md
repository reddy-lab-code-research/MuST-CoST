# MuST-CoST
Code and data for AAAI 2022 paper "Multilingual Code Snippets Training for Program Translation"

## Code
The code is adapted from https://github.com/facebookresearch/CodeGen

### Train the model
First get the checkpoint
'''
wget https://dl.fbaipublicfiles.com/transcoder/pre_trained_models/dobf_plus_denoising.pth
'''


## Data
In CoST_data.zip, there are two folders, raw_data and processed_data. The raw_data contains the .csv files of the programming problems, where each file contains the aligned snippets and programs from different language. The processed_data contains tokenized data that has been splitted into train, validation and test sets.

## Citation
Ming Zhu, Karthik Suresh and Chandan K. Reddy, "Multilingual Code Snippets Training for Program Translation". Proceedings of the 36th AAAI Conference on Artificial Intelligence (AAAI), Feb 22- Mar 1, 2022. Acceptance Rate 15%.
```
@article{zhu2022multilingual,
  title={Multilingual Code Snippets Training for Program Translation},
  author={Zhu, Ming and Suresh, Karthik and Reddy, Chandan K},
  year={2022}
}
```
