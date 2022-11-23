import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset

#load Dataset
dataset_train = load_dataset("mlsum", "de", split='train')
dataset_test = load_dataset("mlsum", "de", split='test')
dataset_eval = load_dataset("mlsum", "de", split='validation')

#Filtern des Datensatzes
train = dataset_train.remove_columns(["date", "summary", "title", "topic", "url"])
test = dataset_test.remove_columns(["date", "summary", "title", "topic", "url"])
evaluation = dataset_eval.remove_columns(["date", "summary", "title", "topic", "url"])

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2", max_len=512)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(examples):

    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets_train = train.map(tokenize_function, batched=True, remove_columns=['text'])
tokenized_datasets_test = test.map(tokenize_function, batched=True, remove_columns=['text'])


from transformers import EarlyStoppingCallback

model = AutoModelForCausalLM.from_pretrained("dbmdz/german-gpt2",
                                             use_cache=False,
                                             pad_token_id=tokenizer.pad_token_id,
                                             )
model.resize_token_embeddings(len(tokenizer))

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./gpt2-news", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    evaluation_strategy ='steps',
    save_total_limit = 5,
    num_train_epochs=3, # number of training epochs
    per_device_train_batch_size=8, # batch size for training
    per_device_eval_batch_size=8,  # batch size for evaluation
    eval_steps = 800, # Number of update steps between two evaluations.
    save_steps=800, # after # steps model is saved
    warmup_steps=500,# number of warmup steps for learning rate scheduler
    )


from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, mlm_probability=0.15
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets_train,
    eval_dataset=tokenized_datasets_test,
    )


trainer.train()

trainer.save_model()


