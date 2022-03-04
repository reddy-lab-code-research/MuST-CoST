from tokenization_utils import *

snippet_data_path = data_path + "snippet_data/"
program_data_path = data_path + "program_data/"
binarize(snippet_data_path, 
         file_extensions.keys(), "data/bpe/cpp-java-python/vocab")
binarize(program_data_path, 
         file_extensions.keys(), "data/bpe/cpp-java-python/vocab")