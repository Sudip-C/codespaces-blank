import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(

        self,

        token_ids,

        max_seq_len

    ):

        self.token_ids = token_ids

        self.max_seq_len = max_seq_len


    def __len__(self):

        return (

            len(self.token_ids)

            - self.max_seq_len

        ) // self.max_seq_len


    def __getitem__(

        self,

        index

    ):

        start = (

            index

            * self.max_seq_len

        )


        input_ids = self.token_ids[

            start:

            start + self.max_seq_len

        ]


        target_ids = self.token_ids[

            start + 1:

            start + self.max_seq_len + 1

        ]


        return (

            torch.tensor(

                input_ids,

                dtype=torch.long

            ),

            torch.tensor(

                target_ids,

                dtype=torch.long

            )

        )