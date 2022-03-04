from codegen_sources.model.translate import *
from codegen_sources.model.preprocess import *
import shutil
from shutil import copyfile
import os 

def get_train_commands(langs, model_path, dump_path, data_path_prefix, exp_prefix, model_exp_prefix,
                       max_epoch=200, max_len=400, beam_size=10,
                       is_reloaded=True, is_dobf=True, is_transferred=False, is_print=True, 
                       is_cont=False, precont_topk=0, cont_lambda=0, max_tok_num=20):
    for lang1 in langs:
        for lang2 in langs:
            if lang2 == lang1:
                continue
            if is_print:
                print("==============", lang1, lang2, "=============")
            exp_name = exp_prefix + lang1 + '_' + lang2
            data_path = data_path_prefix + lang1 + '-' + lang2 + "/"
            if not os.path.exists(data_path):
                data_path = data_path_prefix + lang2 + '-' + lang1 + "/"

            lang_pair = lang_lower[lang1] + "_sa-" + lang_lower[lang2] + "_sa"
            mt_steps = lang_pair
            validation_metrics = "valid_" + lang_pair + "_mt_bleu"
            lgs = lang_pair
            exp_path_prefix = dump_path + exp_name + "/"

            train_command = "python train.py " + \
            "--exp_name " + exp_name + " " + \
            "--dump_path " +  dump_path + " " + \
            "--data_path " + data_path + " " + \
            "--mt_steps " + mt_steps + " " + \
            "--encoder_only False " + \
            "--n_layers 0  " + \
            "--lgs " + lgs  + " " + \
            "--max_vocab 64000 " + \
            "--gelu_activation true " + \
            "--roberta_mode false   " + \
            "--amp 2  " + \
            "--fp16 true  " + \
            "--tokens_per_batch 3000  " + \
            "--group_by_size true " + \
            "--max_batch_size 128  " +  \
            "--epoch_size 10000  " +   \
            "--split_data_accross_gpu global  " +  \
            "--has_sentences_ids true  " + \
            "--optimizer 'adam_inverse_sqrt,warmup_updates=10000,lr=0.0001,weight_decay=0.01'  " + \
            "--eval_bleu true   " + \
            "--eval_computation false   " + \
            "--generate_hypothesis true   " + \
            "--validation_metrics " +  validation_metrics + " " + \
            "--eval_only false" + " " \
            "--max_epoch " + str(max_epoch) + " " +  \
            "--beam_size " + str(beam_size) + " " + \
            "--max_len " + str(max_len) + " "
            if is_dobf:
                train_command += "--n_layers_encoder 12   " + \
                                    "--n_layers_decoder 6  " + \
                                    "--emb_dim 768   " + \
                                    "--n_heads 12   "
            else:
                train_command += "--n_layers_encoder 6   " + \
                                    "--n_layers_decoder 6  " + \
                                    "--emb_dim 1024   "
            if is_reloaded:
                if is_transferred:
                    # Continue training on another checkpoint

                    model_exp_name = model_exp_prefix + lang1 + '_' + lang2
                    model_exp_path_prefix = dump_path + model_exp_name + "/"
                    if not os.path.exists(model_exp_path_prefix):
                        continue
                    exp_key = check_exp_path(model_exp_path_prefix, lang_pair)
                    model_exp_path = model_exp_path_prefix + exp_key + '/'
                    
                    model_path = model_exp_path + "best-valid_" + lang_pair + "_mt_bleu.pth"
                    if not os.path.exists(model_path):
                        print(model_path)
                        continue
                train_command += "--reload_model " + model_path + ',' + model_path + " "
                if is_cont:
                    train_command += "--eval_only True --conditional_generation True  --precont_topk " + \
                                        str(precont_topk) + " --cont_lambda " + str(cont_lambda) +\
                                        " --max_tok_num " + str(max_tok_num)

            print(train_command)
    return

def get_tokenizer(lang):
    processor = LangProcessor.processors[lang](root_folder=so_path)
    tokenizer = processor.tokenize_code
    return tokenizer

def get_detokenizer(lang):
    processor = LangProcessor.processors[lang](root_folder=so_path)
    tokenizer = processor.detokenize_code
    return tokenizer

def get_bpe(is_roberta=False):
    bpe_model = FastBPEMode(codes=os.path.abspath(Fast_codes), vocab_path=None)
    dico = Dictionary.read_vocab(Fast_vocab)
    if is_roberta:
        bpe_model = RobertaBPEMode()
        dico = Dictionary.read_vocab(Roberta_BPE_path)
    return bpe_model, dico

def binarize(root, langs, voc_path):
    iterated_set = set()
    for lang1 in langs:
        for lang2 in langs:
            if lang2 == lang1:
                continue
            if (lang2, lang1) in iterated_set:
                continue
            iterated_set.add((lang1, lang2))
            print(lang1, lang2)
            lang1_lower = lang_lower[lang1]
            lang2_lower = lang_lower[lang2]

            path = root + lang1 + '-' + lang2 + '/'
            fns = os.listdir(path)
            for fn in fns:
                if fn.endswith(".pth"):
                    os.remove(os.path.join(path, fn))
            for tag in tags:
                fn_prefix = path + tag + "-" + lang1 + "-" + lang2
                fn1 = fn_prefix + file_extensions[lang1]
                fn2 = fn_prefix + file_extensions[lang2]
                if tag != 'train':
                    fn1 = fn_prefix + "-trans" + file_extensions[lang1]
                    fn2 = fn_prefix + "-trans" + file_extensions[lang2]
                tag_pth = tag
                if tag == 'val':
                    tag_pth = 'valid'
                fn_pth_prefix = path + tag_pth + "." + lang1_lower + "_sa-" + lang2_lower + "_sa."
                fn1_pth = fn_pth_prefix + lang1_lower + "_sa.pth"
                fn2_pth = fn_pth_prefix + lang2_lower + "_sa.pth"
                fn_pth_prefix_alt = path + tag_pth + "." + lang2_lower + "_sa-" + lang1_lower + "_sa."
                fn1_pth_alt = fn_pth_prefix_alt + lang1_lower + "_sa.pth"
                fn2_pth_alt = fn_pth_prefix_alt + lang2_lower + "_sa.pth"
                print(fn1, fn1_pth, fn2, fn2_pth)
                XLM_preprocess(voc_path, fn1, fn1_pth)
                XLM_preprocess(voc_path, fn2, fn2_pth)
                copyfile(fn1_pth, fn1_pth_alt)
                copyfile(fn2_pth, fn2_pth_alt)
    return


home_path = os.path.dirname(os.path.realpath(__file__)) + "/"
so_path = home_path + "codegen_sources/preprocessing/lang_processors"
Fast_BPE_path = home_path + "data/bpe/cpp-java-python/"
Fast_codes = Fast_BPE_path + 'codes'
Fast_vocab = Fast_BPE_path + 'vocab'
Roberta_BPE_path = home_path + "data/bpe/roberta-base-vocab"
data_path = home_path + "CoST_data_release/processed_data/"
map_data_path = data_path + "map_data/"
snippet_data_path = data_path + "snippet_data/"
program_data_path = data_path + "program_data/"
evaluator_path = home_path + "CodeXGLUE/Code-Code/code-to-code-trans/evaluator/"
split_dict_path = map_data_path + "split_dict.json"

dump_path = home_path + "CodeGen/dumppath1/"
file_extensions = {"Java": ".java", "C++": ".cpp", "C": ".c", "Python": ".py","Javascript": ".js",
                   "PHP":".php", "C#":".cs"}
lang_lower = {"Java": "java", "C++": "cpp", "C": "c", "Python": "python","Javascript": "javascript",
                   "PHP":"php", "C#":"csharp"}
lang_map = {"Java": "java", "C++": "cpp", "C": "c", "Python": "python","Javascript": "javascript",
                   "PHP":"php", "C#":"c_sharp"}
lang_upper = {"java": "Java", "cpp": "C++", "c": "C", "python": "Python","javascript": "Javascript",
                   "php":"PHP", "csharp":"C#"}
tags = ['train', 'val', 'test']

lang_py = 'python'
lang_java = 'java'
lang_cs = 'csharp'
lang_cpp = 'cpp'
lang_c = 'c'
lang_php = 'php'
lang_js = 'javascript'
bpe_model, dico = get_bpe()
py_tokenizer = get_tokenizer(lang_py)
cs_tokenizer = get_tokenizer(lang_cs)
java_tokenizer = get_tokenizer(lang_java)
cpp_tokenizer = get_tokenizer(lang_cpp)
js_tokenizer = get_tokenizer(lang_js)
c_tokenizer = get_tokenizer(lang_c)
# php_tokenizer = get_tokenizer(lang_php)
php_tokenizer = c_tokenizer

py_detokenizer = get_detokenizer(lang_py)
cs_detokenizer = get_detokenizer(lang_cs)
java_detokenizer = get_detokenizer(lang_java)
cpp_detokenizer = get_detokenizer(lang_cpp)
js_detokenizer = get_detokenizer(lang_js)
c_detokenizer = get_detokenizer(lang_c)
# php_tokenizer = get_detokenizer(lang_php)
php_detokenizer = c_detokenizer

file_tokenizers = {"Java": java_tokenizer, "C++": cpp_tokenizer, "C": c_tokenizer, "Python": py_tokenizer,
                   "Javascript": js_tokenizer, "PHP": php_tokenizer, "C#": cs_tokenizer}
file_detokenizers = {"Java": java_detokenizer, "C++": cpp_detokenizer, "C": c_detokenizer, "Python": py_detokenizer,
                   "Javascript": js_detokenizer, "PHP": php_detokenizer, "C#": cs_detokenizer}
langs = ["C++", "Java", "Python", "C#", "Javascript", "PHP", "C"]