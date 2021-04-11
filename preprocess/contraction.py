"""
according to the official BERT repo:
There are common English tokenization schemes which will cause a slight mismatch between how BERT was pre-trained.
For example, if your input tokenization splits off contractions like do n't, this will cause a mismatch.
If it is possible to do so, you should pre-process your data to convert these back to raw-looking text
"""

from os import path, listdir
import re

# whether the outputs are in a single file
AGGREGATED = False
KEYWORD = 'full'
# KEYWORD = 'coach'

# DIR = path.join('C:/Users/Mizuk/Documents/phD/csc2611/COHA/filtered/', KEYWORD)
# input_dir = path.join(DIR, 'separated')
DIR = 'C:/Users/Mizuk/Documents/BERT/after_model/full_coha/pickle/human_sim/'
input_dir = path.join(DIR, 'before/')

output_list = []
for filename in listdir(input_dir):
    input_path = path.join(input_dir, filename)
    if not path.isfile(input_path):
        continue
    print("processing file: {}".format(input_path))
    output_path = path.join(DIR, 'full_processed', 'full_inputs_10.txt')
    output_path = 'trivial'
    if AGGREGATED:
        with open(input_path, "r") as r:
            print("writing to {}".format(output_path))
            with open(output_path, "a") as w:
                w.write(r.read().replace(" n't", "n't"))
                w.write('\n\n')
    else:
        output_path = path.join(DIR, 'processed', filename)
        print("writing to {}".format(output_path))
        with open(input_path, "r") as r:
            with open(output_path, "w") as w:
                text = r.read()
                print("raw: ")
                print(text)
                text = re.sub(r'\s([?,\'.:!/)%"](?:\s|$))', r'\1', text)
                print("re: ")
                print(text)
                text = text.replace("' ", "'").replace("( ", "(").replace(") .", ").").replace(" - - ", "--").\
                    replace(" - ", "-")
                print("final: ")
                print(text)
                w.write(text.replace(" n't", "n't"))

