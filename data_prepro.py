from tokenization_utils import *


binarize(snippet_data_path, 
         file_extensions.keys(), "data/bpe/cpp-java-python/vocab")
binarize(program_data_path, 
         file_extensions.keys(), "data/bpe/cpp-java-python/vocab")