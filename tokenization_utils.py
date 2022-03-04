from codegen_sources.model.translate import *
from codegen_sources.model.preprocess import *
import shutil
from shutil import copyfile

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

                
home_path = "./"
so_path = home_path + "codegen_sources/preprocessing/lang_processors"
Fast_BPE_path = home_path + "data/bpe/cpp-java-python/"
Fast_codes = Fast_BPE_path + 'codes'
Fast_vocab = Fast_BPE_path + 'vocab'
Roberta_BPE_path = home_path + "data/bpe/roberta-base-vocab"
data_path = home_path + "CoST_data_release/processed_data/"
map_data_path = data_path + "map_data/"
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