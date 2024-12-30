# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from importlib.metadata import version


import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)  # A

        for i in range(0, len(token_ids) - max_length, stride):  # B
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):  # C
        return len(self.input_ids)

    def __getitem__(self, idx):  # D
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4,
        max_length=256, stride=128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2") #A
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride) #B
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last) #C
    return dataloader

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('AI2')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

with open("the-verdict.txt", "r", encoding="utf-8") as f:
   raw_text = f.read()
# print("Total number of character:", len(raw_text))
#
# print ("tiktoken version:", version("tiktoken"))
#
# tokenizer1 = tiktoken.get_encoding("gpt2")
#
#
# enc_text = tokenizer1.encode(raw_text)
# print(len(enc_text))
#
# enc_sample = enc_text[50:]
#
# context_size = 4 #A
#
# x = enc_sample[:context_size]
# y = enc_sample[1:context_size+1]
# print(f"x: {x}")
# print(f"y:      {y}")
#
# for i in range(1, context_size+1):
#     context = enc_sample[:i]
#     desired = enc_sample[i]
#     print(context, "---->", desired)
#
# for i in range(1, context_size+1):
#     context = enc_sample[:i]
#     desired = enc_sample[i]
#     print(tokenizer1.decode(context), "---->", tokenizer1.decode([desired]))

dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader) #A
first_batch = next(data_iter)
print(first_batch)

second_batch = next(data_iter)
print(second_batch)

third_batch = next(data_iter)
print(third_batch)

