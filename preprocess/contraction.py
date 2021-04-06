"""
according to the official BERT repo:
There are common English tokenization schemes which will cause a slight mismatch between how BERT was pre-trained.
For example, if your input tokenization splits off contractions like do n't, this will cause a mismatch.
If it is possible to do so, you should pre-process your data to convert these back to raw-looking text
"""

from os import path, listdir

# whether the outputs are in a single file
AGGREGATED = True
KEYWORD = 'full'
# KEYWORD = 'coach'

DIR = path.join('C:/Users/Mizuk/Documents/phD/csc2611/COHA/filtered/', KEYWORD)
input_dir = path.join(DIR, 'separated')

output_list = []
for filename in listdir(input_dir):
    input_path = path.join(input_dir, filename)
    output_path = path.join(DIR, 'full_processed', 'full_inputs.txt')
    if AGGREGATED:
        with open(input_path, "r") as r:
            with open(output_path, "a") as w:
                w.write(r.read().replace(" n't", "n't"))
                w.write('\n\n')
    else:
        output_path = path.join(DIR, 'processed', filename)
        with open(input_path, "r") as r:
            with open(output_path, "w") as w:
                w.write(r.read().replace(" n't", "n't"))

