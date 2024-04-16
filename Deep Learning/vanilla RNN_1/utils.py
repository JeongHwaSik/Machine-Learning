#!/usr/bin/env python
# coding: utf-8

# In[74]:


import os
import io
import glob
import unicodedata
import string
import torch
import random

# All the ASCII letters including ".,;'"
# abcdefghijklmnopqrstuvwxyzABCDEFJHIJKLMNOPQRSTUVWXYZ.,;'
ALL_LETTERS = string.ascii_letters + ".,;'"
N_LETTERS = len(ALL_LETTERS) # 56

# to remove all the accents
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in ALL_LETTERS)

def load_data():
    all_categories = []
    category_lines = {}
    
    # Find all files
    def find_files(path):
        return glob.glob(path)

    # Read a file and split it into a list
    def read_lines(filename):
        lines = io.open(filename, mode='r', encoding='utf-8').read().strip().split('\n')
        return [unicode_to_ascii(line) for line in lines]

    for filename in find_files('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        
        lines = read_lines(filename)
        category_lines[category] = lines
    
    return all_categories, category_lines


def letter_to_index(letter):
    return ALL_LETTERS.find(letter)

def letter_to_tensor(letter):
    # Make a letter into one-hot-encoding vector
    # For example, the size of the tensor for a letter "a" will be (1,56)
    tensor = torch.zeros((1, N_LETTERS))
    tensor[0][letter_to_index(letter)] = 1
    return tensor

def line_to_tensor(line):
    # Make a line into tensor
    # For example, the size of the tensor for a word "Korean" will be (6, 1, 56)
    tensor = torch.zeros((len(line), 1, N_LETTERS))
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor

def random_training_example(all_categories, category_lines):
    random_index = random.randint(0, len(all_categories)-1)
    category = all_categories[random_index]
    one_word = random.choice(category_lines[category])
    
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    word_tensor = line_to_tensor(one_word)
    return category, one_word, category_tensor, word_tensor


if __name__ == '__main__':
    print(ALL_LETTERS)
    print(unicode_to_ascii('Ślusàrski'))
    
    all_categories, category_lines = load_data()
    print(category_lines['Korean'][:5])
    
    print(letter_to_tensor('a'))
    print(letter_to_tensor('a').size())
    print(line_to_tensor('Korean'))
    print(line_to_tensor('Korean').size())

