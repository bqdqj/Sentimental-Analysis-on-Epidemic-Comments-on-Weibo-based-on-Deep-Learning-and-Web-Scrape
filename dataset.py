from torchtext import data
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchtext.vocab import Vectors
from torchtext.data import Iterator, BucketIterator


class MyDataset(data.Dataset):

    def __init__(self, trainset, label, id_num, text_field, label_field, id_field=None, test=False, **kwargs):
        fields = [("id", id_field), ('text', text_field), ("label", label_field)]

        examples = []
        dataset = trainset
        label = label
        id_num = id_num

        if test:
            for text, id_num in zip(dataset, id_num):
                examples.append(data.Example.fromlist([id_num, text, None], fields))
        else:
            for text, label in zip(dataset, label):
                examples.append(data.Example.fromlist([None, text, label], fields))

        super(MyDataset, self).__init__(examples, fields, **kwargs)