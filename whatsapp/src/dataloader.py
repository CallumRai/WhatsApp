from torch.utils.data import Dataset, random_split, RandomSampler
import torch
from transformers import GPT2Tokenizer
import os
import pickle


class GPT2Dataset(Dataset):
    """
    Class for a dataset of encoded strings that is compatible with GPT2 and PyTorch

    Attributes:
        input_ids: list[torch.tensor(int)]
            Contains each string encoded as integers word-wise
        attn_masks: list[torch.tensor(int)]
            1 if corresponding id is a word, 0 if is padding
    """

    def __init__(self, txt_list, max_length=768):
        """
        Creates dataset

        Args:
            txt_list: list[str]
                Strings to put into dataset
            max_length: Optional int
                Number of words to truncate/pad string to
        """
        super().__init__()

        self.input_ids = []
        self.attn_masks = []

        # Define tokeniszer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>',
                                                  pad_token='<|pad|>')
        for txt in txt_list:
            # Add start and end tokens
            encodings_dict = tokenizer('<|startoftext|> ' + txt + ' <|endoftext|>', truncation=True,
                                           max_length=max_length, padding="max_length")

            # Text tokenized
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))

            # Attention mask (0 if padding)
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


class DataLoader:
    """
    Class for creating dataloaders for encoded strings compatible with GPT2 and PyTorch

    Attributes:
        contact: str
            Name of contact to make a corpus from
        file_name: str (Optional)
            Name of file where WhatsApp text is saved, defaults to _chat
    """

    def __init__(self, contact, file_name = "_chat"):
        """
        Initialise dataloader class

        Args:
            contact: str
                Name of contact to make a corpus from
            file_name: str (Optional)
                Name of file where WhatsApp text is saved, defaults to _chat
        """

        self.contact = contact
        self.file_name = file_name

    def validate(self, train_prop, batch_size):
        """
        Creates validation and training dataloaders to validate model with

        Args:
            train_prop: float
                Proportion of data to train upon (as oppose to evaluate upon)
            batch_size: int
                Size of batches to train on

        Returns:
            train_dataloader: torch.utils.data.DataLoader
            val_dataloader: torch.utils.data.DataLoader
        """

        corpus_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + "/data/corpus/"
        f_name = f"{self.file_name}_{self.contact}.txt"

        # Loads corpus as list
        f = open(corpus_path + f_name, "rb")
        corpus = pickle.load(f)
        f.close()

        # Filter out empty strings
        corpus = list(filter(None, corpus))

        # Creates full dataset
        dataset = GPT2Dataset(corpus, 200)

        # Splits dataset into train and validation sets
        train_size = int(train_prop * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Put datasets into loaders which randomly choose batches
        train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=RandomSampler(train_dataset)
                                                       , batch_size=batch_size)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, sampler=RandomSampler(val_dataset)
                                                     , batch_size=batch_size)

        return train_dataloader, val_dataloader

    def train(self, batch_size):
        """
        Creates training dataloader to debug model with

        Args:
            batch_size: int
                Size of batches to train on

        Returns:
            train_dataloader: torch.utils.data.DataLoader
        """

        corpus_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..')) + "/data/corpus/"
        f_name = f"{self.file_name}_{self.contact}.txt"

        # Loads corpus as list
        f = open(corpus_path + f_name, encoding="utf-8")
        corpus = f.read().splitlines()
        f.close()

        # Filter out empty strings
        corpus = list(filter(None, corpus))

        # Creates full dataset
        train_dataset = GPT2Dataset(corpus, 200)

        # Put datasets into loaders which randomly choose batches
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       sampler=RandomSampler(
                                                           train_dataset)
                                                       , batch_size=batch_size)

        return train_dataloader