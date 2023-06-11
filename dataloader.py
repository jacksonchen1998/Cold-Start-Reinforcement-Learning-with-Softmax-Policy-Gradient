import torch
import random
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from collections import Counter
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

dataset = load_dataset("gigaword")
train_set = dataset['train']
<<<<<<< HEAD
=======
train_set = dataset['test']
>>>>>>> 02163493d27dc05f21316949c03f10b9ebaa1996
val_set = dataset['validation']
test_set = dataset['test']

train_set = test_set

# print(len(train_set), len(val_set), len(test_set))

# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer = get_tokenizer('basic_english')

def counter_tokens(dataset, col_name):
    counter = Counter()
    # set_name = ['train', 'validation', 'test']
    # for set in set_name:
    #     data_iter = dataset[set]
    for data in dataset:
        col = data[col_name]
        counter.update(tokenizer(col))
    return counter

# vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>"])
SRC_vocab = vocab(counter_tokens(train_set, 'document'), min_freq = 2, specials=('<unk>', '<BOS>', '<EOS>', '<PAD>'))
TRG_vocab = vocab(counter_tokens(train_set, 'summary'), min_freq = 2, specials=('<unk>', '<BOS>', '<EOS>', '<PAD>'))
SRC_vocab.set_default_index(SRC_vocab['<unk>'])
TRG_vocab.set_default_index(TRG_vocab['<unk>'])

# text_transform = lambda x: [SRC_vocab['<BOS>']] + [SRC_vocab[token] for token in tokenizer(x)] + [SRC_vocab['<EOS>']]
def text_transform(text, vocab):
    return [vocab['<BOS>']] + [vocab[token] for token in tokenizer(text)] + [vocab['<EOS>']]

def get_vocab():
    return SRC_vocab, TRG_vocab

# print("The length of the 'document' vocab is", len(SRC_vocab))
# print("The length of the 'summary' vocab is", len(TRG_vocab))
# new_stoi = SRC_vocab.get_stoi()
# print("The index of '<PAD>' is", new_stoi['<PAD>'])
# new_itos = SRC_vocab.get_itos()
# print("The token at index 1 is", new_itos[1])
# lambda x: [vocab['<BOS>']] + [vocab[token] for token in tokenizer(x)] + [vocab['<EOS>']]
# trg_transform = lambda x: 1 if x == 'pos' else 0

# Print out the output of text_transform
# print("input to the text_transform:", "here is an example")
# print("output of the text_transform:", text_transform("here is an example"))

# train_list = list(train_set)

def collate_batch(batch):
   target_list, src_list = [], []
   for data in batch:
      _summary, _document = data['summary'], data['document']
      target_text = torch.tensor(text_transform(_summary, TRG_vocab))
      src_text = torch.tensor(text_transform(_document, SRC_vocab))
      target_list.append(target_text)
      src_list.append(src_text)
   return pad_sequence(src_list, padding_value=3), pad_sequence(target_list, padding_value=3)

def get_train_dataloader(batch_size):
    train_dataloader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_batch)
    return train_dataloader

batch_size = 8  # A batch size of 8

def batch_sampler(train_list):
    indices = [(i, len(tokenizer(s['document']))) for i, s in enumerate(train_list)]
    random.shuffle(indices)
    pooled_indices = []
    # create pool of indices with similar lengths 
    for i in range(0, len(indices), batch_size * 100):
        pooled_indices.extend(sorted(indices[i:i + batch_size * 100], key=lambda x: x[1]))

    pooled_indices = [x[0] for x in pooled_indices]

    # yield indices for current batch
    for i in range(0, len(pooled_indices), batch_size):
        yield pooled_indices[i:i + batch_size]

# bucket_dataloader = DataLoader(train_list, batch_sampler=batch_sampler(),
                            #    collate_fn=collate_batch)

class GigawordDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataset
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        text = data['document']
        summary = data['summary']

        inputs = self.tokenizer.encode_plus(
            text,
            summary,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'summary_ids': inputs['labels'].flatten()
        }
    
# train_dataset = GigawordDataset(train_set, tokenizer, 512)