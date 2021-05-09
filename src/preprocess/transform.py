import os
import random
import json
import xml.etree.ElementTree as et
from cfg import *

random.seed(42)

label_dict = {
    'neutral': 0,
    'positive': 1,
    'negative': 2
}

def transform_asp(raw_path, out_path):
    parser = et.parse(raw_path)
    with open(out_path, 'w', encoding='utf-8') as f:
        root = parser.getroot()
        for sentence in root.iter('sentence'):
            text = sentence.find('text').text
            text_list = text.split(' ')
            aspects = sentence.find('aspectTerms')
            if aspects is None:
                continue
            for aspect in aspects.iter('aspectTerm'):
                term = aspect.get('term')
                polar = aspect.get('polarity')
                if polar == 'conflict':
                    continue
                char_start = int(aspect.get('from'))
                # char_end = int(aspect.get('to'))
                term_list = term.split(' ')
                left_text = text[:char_start]
                start = len(left_text.strip().split(' '))
                end = start + len(term_list)
                print(term, polar, start, end)
                sample = {
                    'text': text_list,
                    'aspect': [term, label_dict[polar], start, end]
                }
                f.write(json.dumps(sample) + '\n')

def split(data_path, train_path, dev_path, percent=0.2):
    train_f = open(train_path, 'w', encoding='utf-8')
    dev_f = open(dev_path, 'w', encoding='utf-8')
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        dev_idxs = random.sample(range(len(lines)), int(percent*len(lines)))
        for idx, line in enumerate(lines):
            if idx in dev_idxs:
                dev_f.write(line)
            else:
                train_f.write(line)

if __name__ == '__main__':
    # transform_asp(os.path.join(DATA_PATH, 'train.xml'), os.path.join(DATA_PATH, 'data.json'))
    # transform_asp(os.path.join(DATA_PATH, 'dev.xml'), os.path.join(DATA_PATH, 'dev.json'))
    transform_asp(os.path.join(DATA_PATH, 'test.xml'), os.path.join(DATA_PATH, 'test.json'))
    # split(os.path.join(DATA_PATH, 'data.json'), os.path.join(DATA_PATH, 'train.json'), os.path.join(DATA_PATH, 'dev.json'))
