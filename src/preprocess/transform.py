import os
import re
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
            RE_TEMPLATE = r"[\w']+|[.,!?;]"
            text = sentence.find('text').text
            text = text.replace('\u00a0', '')
            text_list = re.findall(RE_TEMPLATE, text)
            aspects = sentence.find('aspectTerms')
            ate_label = ['O' for _ in range(len(text_list))]
            sentence_aspects = []
            # TODO remove below
            if aspects is None:
                if (sentence_aspects == []):
                    print("NO")
                sentence_sample = {
                    'id': sentence.get('id'),
                    'text': text_list,
                    'aspects': [],
                    'label': ['O'] * len(text_list)
                }
                f.write(json.dumps(sentence_sample) + '\n')
                continue
            for aspect in aspects.iter('aspectTerm'):
                term = aspect.get('term')
                polar = aspect.get('polarity')
                if polar == 'conflict':
                    continue
                char_start = int(aspect.get('from'))
                # term_list = term.split()
                term_list = re.findall(RE_TEMPLATE, term)
                left_text = text[:char_start]
                start = len(re.findall(RE_TEMPLATE, left_text.strip())) if len(left_text) > 0 else 0
                end = start + len(term_list)
                ate_label[start] = "B"
                for t in range(start + 1, end):
                    ate_label[t] = "I"
                print(term, polar, start, end)
                sentence_aspects.append([term, label_dict[polar], start, end])
            sentence_sample = {
                'id': sentence.get('id'),
                'text': text_list,
                'aspects': sentence_aspects,
                'label': ate_label
            }
            if (sentence_aspects == []):
                continue
            f.write(json.dumps(sentence_sample) + '\n')

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

