from tokenization_utils import *

# Manage experiments
model_path = home_path + "dobf_plus_denoising.pth"  
dump_path = home_path + "dumppath1/"
snippet_data_path = data_path + "snippet_data/"
program_data_path = data_path + "program_data/"
prefix = "dobf_"


# Train dobf on snippets
exp_data_path = snippet_data_path
exp_prefix = prefix + "snippet_"
model_exp_prefix = ""
max_epoch = 50
max_len = 100
beam_size = 5
print("Snippet Translation Commands")
get_train_commands(langs, model_path, dump_path, exp_data_path, exp_prefix, model_exp_prefix,
                       max_epoch=max_epoch, max_len=max_len, beam_size=beam_size,
                       is_reloaded=True, is_dobf=True, is_transferred=False, is_print=True)

# Train dobf on program
exp_data_path = program_data_path
exp_prefix = prefix + "program_"
model_exp_prefix = ""
max_epoch = 50
max_len = 400
beam_size = 5
print("Program Translation Commands")
get_train_commands(langs, model_path, dump_path, exp_data_path, exp_prefix, model_exp_prefix,
                       max_epoch=max_epoch, max_len=max_len, beam_size=beam_size,
                       is_reloaded=True, is_dobf=True, is_transferred=False, is_print=True)

# Train dobf on program with pre-trained snippet model
exp_data_path = program_data_path
exp_prefix = prefix + "program_transfer_"
model_exp_prefix = prefix + "snippet_"
max_epoch = 50
max_len = 400
beam_size = 5
print("MuST Commands")
get_train_commands(langs, model_path, dump_path, exp_data_path, exp_prefix, model_exp_prefix,
                       max_epoch=max_epoch, max_len=max_len, beam_size=beam_size,
                       is_reloaded=True, is_dobf=True, is_transferred=True, is_print=True)


