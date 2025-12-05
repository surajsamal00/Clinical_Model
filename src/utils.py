from torch.utils.data import Dataset

class ClinicalDataset(Dataset):
    def __init__(self, df, tokenizer, max_input_len=256, max_output_len=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Encoder input (findings)
        inputs = self.tokenizer(
            row["findings"],
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Decoder target (impression)
        targets = self.tokenizer(
            row["impression"],
            max_length=self.max_output_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze()
        }