import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset
from torch.utils.data import DataLoader

def load_data_split(tokenizer: AutoTokenizer, 
                    dataset: pd.DataFrame,
                    batch_size: int = 64) -> DataLoader:
    # Read CSV file
    # df = dataset_prep(pd.read_csv(dataset_path))
    # # Create Dataset object
    ds = Dataset.from_pandas(dataset)
    # Define tokenizer function
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)
    # Tokenize values in Dataset object
    ds = ds.map(tokenize_function, batched=True)
    # Remove original text from dataset
    ds = ds.remove_columns("text")
    # Create DataCollator object
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # Create DataLoader to be returned
    return DataLoader(dataset=ds, 
                    batch_size=batch_size, 
                    collate_fn=data_collator)

def extract_features(model: AutoModel, data_loader: DataLoader) -> np.ndarray:
   last_hidden_states = torch.tensor([]).to(model.device)
   with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            print(f'Running inference with batch no. {idx+1}')
            batch = {k: v.to(model.device) for k, v in batch.items()} 
            last_hidden_states_for_batch = model(**batch) 
            last_hidden_states = torch.cat((last_hidden_states, last_hidden_states_for_batch[0].to(model.device)))

        # extract [CLS] token hidden representation from output layer
        return last_hidden_states[:,0,:].cpu().numpy()
