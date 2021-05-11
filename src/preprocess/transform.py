import os
import random
import json
import xml.etree.ElementTree as et
from cfg import *

random.seed(41)

label_dict = {
    'neutral': 0,
    'positive': 1,
    'negative': 2
}

def get_test_id(test_path):
    test_ids = set()
    parser = et.parse(test_path)
    root = parser.getroot()
    for sentence in root.iter('sentence'):
        test_ids.add(sentence.get('id'))
    return test_ids


def transform_asp(raw_path, out_path):
    parser = et.parse(raw_path)
    with open(out_path, 'w', encoding='utf-8') as f:
        root = parser.getroot()
        for sentence in root.iter('sentence'):
            text = sentence.find('text').text
            text = text.replace('\u00a0', '')
            text_list = text.split()
            aspects = sentence.find('aspectTerms')
            if aspects is None:
                continue
            for aspect in aspects.iter('aspectTerm'):
                term = aspect.get('term')
                polar = aspect.get('polarity')
                if polar == 'conflict':
                    continue
                char_start = int(aspect.get('from'))
                term_list = term.split()
                left_text = text[:char_start]
                start = len(left_text.strip().split()) if len(left_text) > 0 else 0
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

def get_json_data(dataset_name):
    transform_asp(os.path.join(DATA_PATH, dataset_name, 'train.xml'), os.path.join(DATA_PATH, dataset_name, 'data.json'))
    split(os.path.join(DATA_PATH, dataset_name, 'data.json'), os.path.join(DATA_PATH, dataset_name, 'train.json'),
          os.path.join(DATA_PATH, dataset_name, 'dev.json'))
    transform_asp(os.path.join(DATA_PATH, dataset_name, 'test.xml'), os.path.join(DATA_PATH, dataset_name, 'test.json'))

if __name__ == '__main__':
    get_json_data('restaurant')
    get_json_data('laptop')

