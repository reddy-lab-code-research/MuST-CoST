# MuST-CoST
Code and data for AAAI 2022 paper "Multilingual Code Snippets Training for Program Translation"

## Code
The code is adapted from https://github.com/facebookresearch/CodeGen

### Train the model
First get the checkpoint
```
wget https://dl.fbaipublicfiles.com/transcoder/pre_trained_models/dobf_plus_denoising.pth
```
Snippet Translation:
```
python train.py --exp_name exp_snippet_<lang1>_<lang2> --dump_path dumppath1 --data_path CoST_data/snippet_data/<lang1>_<lang2>/ --mt_steps <lang1>_sa-<lang2>_sa --encoder_only False --n_layers 0  --lgs <lang1>_sa-<lang2>_sa --max_vocab 64000 --gelu_activation true --roberta_mode false   --amp 2  --fp16 true  --tokens_per_batch 3000  --group_by_size true --max_batch_size 128  --epoch_size 10000  --split_data_accross_gpu global  --has_sentences_ids true  --optimizer 'adam_inverse_sqrt,warmup_updates=10000,lr=0.0001,weight_decay=0.01'  --eval_bleu true   --eval_computation false   --generate_hypothesis true   --validation_metrics valid_<lang1>_sa-<lang2>_sa_mt_bleu --eval_only false --max_epoch 50 --beam_size 5 --max_len 100 --n_layers_encoder 12   --n_layers_decoder 6  --emb_dim 768   --n_heads 12   --reload_model dobf_plus_denoising.pth,dobf_plus_denoising.pth 
```

Program Translation:
Replace the "snippet_data" with "program_data" in data_path. Note that you may want to change the exp_name accordingly also.

MuST Training:
```
python train.py --exp_name exp_<lang1>_<lang2> --dump_path dumppath1 --data_path CoST_data/<data_type>/<lang1>_<lang2>/ --mt_steps <lang1>_sa-<lang2>_sa --encoder_only False --n_layers 0  --lgs <lang1>_sa-<lang2>_sa --max_vocab 64000 --gelu_activation true --roberta_mode false   --amp 2  --fp16 true  --tokens_per_batch 3000  --group_by_size true --max_batch_size 128  --epoch_size 10000  --split_data_accross_gpu global  --has_sentences_ids true  --optimizer 'adam_inverse_sqrt,warmup_updates=10000,lr=0.0001,weight_decay=0.01'  --eval_bleu true   --eval_computation false   --generate_hypothesis true   --validation_metrics valid_<lang1>_sa-<lang2>_sa_mt_bleu --eval_only false --max_epoch 50 --beam_size 5 --max_len 100 --n_layers_encoder 12   --n_layers_decoder 6  --emb_dim 768   --n_heads 12     --reload_model dumppath1/exp_<lang1>_<lang2>/<exp_id>/best-valid_<lang1>_sa-<lang2>_sa_mt_bleu.pth,exp_<lang1>_<lang2>/<exp_id>/best-valid_<lang1>_sa-<lang2>_sa_mt_bleu.pth
```

DAE pre-training (using Java as an example):
```
python train.py --exp_name all_2_Java --dump_path dumppath1 --data_path all_2_one/Java/ --mt_steps cpp_sa-java_sa,c_sa-java_sa,python_sa-java_sa,javascript_sa-java_sa,php_sa-java_sa,csharp_sa-java_sa --encoder_only False --n_layers 0  --lgs cpp_sa-c_sa-python_sa-javascript_sa-php_sa-csharp_sa-java_sa --max_vocab 64000 --gelu_activation true --roberta_mode false   --amp 2  --fp16 true  --tokens_per_batch 3000  --group_by_size true --max_batch_size 128  --epoch_size 10000  --split_data_accross_gpu global  --has_sentences_ids true  --optimizer 'adam_inverse_sqrt,warmup_updates=10000,lr=0.0001,weight_decay=0.01'  --eval_bleu true   --eval_computation false   --generate_hypothesis true   --validation_metrics valid_cpp_sa-java_sa_mt_bleu,valid_c_sa-java_sa_mt_bleu,valid_python_sa-java_sa_mt_bleu,valid_javascript_sa-java_sa_mt_bleu,valid_php_sa-java_sa_mt_bleu,valid_csharp_sa-java_sa_mt_bleu --eval_only false --max_epoch 200 --beam_size 10 --max_len 100 --ae_steps java_sa --lambda_ae 0:1,30000:0.1,100000:0 --n_layers_encoder 12   --n_layers_decoder 6  --emb_dim 768   --n_heads 12   --reload_model dobf_plus_denoising.pth,dobf_plus_denoising.pth 
```

Evaluation:
Change "--eval_only" in the training command from false to true.



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
