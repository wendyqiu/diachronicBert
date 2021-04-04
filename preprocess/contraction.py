"""
according to the official BERT repo:
There are common English tokenization schemes which will cause a slight mismatch between how BERT was pre-trained.
For example, if your input tokenization splits off contractions like do n't, this will cause a mismatch.
If it is possible to do so, you should pre-process your data to convert these back to raw-looking text
"""

from os import path

# decade/directory number (from 1 to 10)
NUM = 1
current_decade = '19' + str(NUM) + '0'
if current_decade == '19100':
    current_decade = '2000'

DIR = 'C:/Users/Mizuk/Documents/phD/csc2611/COHA/filtered/coach/'

filename = current_decade + '.txt'
input_path = path.join(DIR, current_decade, filename)
output_path = path.join(DIR, 'processed', filename)

with open(input_path, "r") as r:
    with open(output_path, "w") as w:
        w.write(r.read().replace(" n't", "n't"))