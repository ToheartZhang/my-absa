from transformers import RobertaTokenizer, RobertaModel, BertTokenizer

# tokenizer = RobertaTokenizer.from_pretrained('roberta-base', mirror='tuna')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = 'I am a dog.'
list_text = ['I', 'am', 'a', 'dog','.']
e = tokenizer.encode(text)
list_text = ['<s>'] + list_text + ['</s>']
pieces = []
for i, w in enumerate(list_text):
    if i >= 2 and i < len(list_text) - 1:
        bpes = tokenizer.convert_tokens_to_ids(
                                tokenizer.tokenize(' ' + w)
                            )
    else:
        bpes = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(w)
        )
    pieces.extend(bpes)
print(e)
print(pieces)
print(tokenizer.decode(e))
print(tokenizer.decode(pieces))
