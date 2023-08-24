import pandas as pd
import torch
from torch import nn
from transformers import BertForSequenceClassification, BertConfig
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup

from utils.finetune_utils import FineTuneTrainer
from utils.finetune_utils import make_finetune_dataloaders, make_finetune_dataloader

# Preallocate variables defined in set_training_hyperparameters
training_params = dict(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                       epochs=5,
                       batch_size=8,
                       eval_every=1,
                       eval_iters=1,
                       max_seq_len=512,
                       save_every=10000, )

learning_params = dict(lr=3e-4, eps=1e-8)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = training_params['max_seq_len']
batch_size = training_params['batch_size']

folder_loc = 'data/shelfbounce/'
file_loc = 'variable_height'

train_data = pd.read_csv(folder_loc + file_loc + '/train_data.csv')
val_data = pd.read_csv(folder_loc + file_loc + '/val_data.csv')
test_data = pd.read_csv(folder_loc + file_loc + '/test_data.csv')
oos_test_data = pd.read_csv(folder_loc + file_loc + '/oos_test_data.csv')

model_name = 'bert-base-uncased'

# Create the data loaders
train_dataloader, val_dataloader, test_dataloader = make_finetune_dataloaders(train_data=train_data, val_data=val_data,
                                                                              test_data=test_data, tokenizer=tokenizer,
                                                                              max_length=max_length,
                                                                              batch_size=batch_size, output_type='num')

oos_dataloader = make_finetune_dataloader(data=oos_test_data, tokenizer=tokenizer, max_length=max_length,
                                          batch_size=batch_size, output_type='num')

config = BertConfig.from_pretrained(model_name)
model = BertForSequenceClassification(config)
config.num_labels = 1

# Freeze all but the classification layer
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the classification layer
for param in model.classifier.parameters():
    param.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_params['lr'], eps=learning_params['eps'])
epochs = training_params['epochs']
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * epochs)
loss_fn = nn.MSELoss()

BertTrainer = FineTuneTrainer(model=model,
                              optimiser=optimizer,
                              scheduler=scheduler,
                              loss_fn=loss_fn,
                              training_hyperparameters=training_params,
                              tokenizer=tokenizer,
                              )

model, _, _ = BertTrainer.train(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    save_model=True,
    plotting=True,
    verbose=True,
    early_stopping=True,
    early_stopping_patience=10,
)

test_error = BertTrainer.log_numerical_outputs(test_dataloader, output_type='num')
print(f'Test error: {test_error:,.4f}')

oos_test_error = BertTrainer.log_numerical_outputs(oos_dataloader, output_type='num')
print(f'OOS Test error: {oos_test_error:,.4f}')


