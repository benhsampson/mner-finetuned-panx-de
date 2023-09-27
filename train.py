# pylint: disable=redefined-builtin, abstract-method

from typing import Optional, Union
from collections import defaultdict

from rich import print
import numpy as np
import pandas as pd
from datasets import DatasetDict, load_dataset, Dataset, Sequence, ClassLabel
from transformers import AutoTokenizer, XLMRobertaConfig, AutoConfig, BatchEncoding, TrainingArguments, DataCollatorForTokenClassification, Trainer, logging
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
import torch
import torch.nn as nn
from seqeval.metrics import f1_score
from huggingface_hub import login

# LOGIN TO HUGGINGFACE HUB

login()

# PREPARE DATASET
# TODO: cache this dataset

langs = ['de', 'fr', 'it', 'en']
fracs = [0.629, 0.229, 0.084, 0.059]
panx_ch: defaultdict[str, DatasetDict] = defaultdict(DatasetDict)
for lang, frac in zip(langs, fracs):
    ds = load_dataset('xtreme', name=f'PAN-X.{lang}')
    for split in ds:
        ds[split]: Dataset
        panx_ch[lang][split] = (
            ds[split]
            .shuffle(seed=0)
            .select(range(int(frac*ds[split].num_rows))))

ner_tags: Sequence = panx_ch['de']['train'].features['ner_tags']
tags: ClassLabel = ner_tags.feature

# print the number of samples in each split, for each language
print(pd.DataFrame({
   split: [panx_ch[lang][split].num_rows for lang in langs] for split in ['train', 'validation', 'test']
}, index=langs).T)

# PREPARE TOKENIZER

xlmr_model_name = 'xlm-roberta-base'
xlmr_tokenizer = AutoTokenizer.from_pretrained(xlmr_model_name)

with open('quotes.txt', 'r', encoding='utf-8') as f:
    quotes = f.read().splitlines()[:3]
    for q in quotes:
        xlmr_tokens = xlmr_tokenizer.tokenize(q)
        input_tensor: torch.Tensor = xlmr_tokenizer.encode(q, return_tensors='pt')
        input_ids: np.ndarray = input_tensor[0].numpy()
        print(pd.DataFrame([xlmr_tokens, input_ids], index=['tokens', 'input ids']))

# CREATE MODEL

class XLMRobertaForTokenClassification(RobertaPreTrainedModel):
    config_class = XLMRobertaConfig

    def __init__(self, config: config_class):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(
            config,
            # return all hidden states
            #  not just the one associated with the [CLS] token
            add_pooling_layer=False
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # use pretrained weights of roberta body
        #  and randomly initialize weights of classifier head
        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> TokenClassifierOutput:
        # use model body to get encoder output
        outputs = self.roberta.forward(
            input_ids,
            attention_mask,
            token_type_ids,
            **kwargs
        )
        # apply classifier to encoder representation
        sequence_output = self.dropout.forward(
            outputs[0]
        )
        logits = self.classifier.forward(sequence_output)
        # calcuate losses
        loss: Optional[torch.Tensor] = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn.forward(
                logits.view(-1, self.num_labels),
                labels.view(-1)
            )
        # return model output object
        return TokenClassifierOutput(
            loss,
            logits,
            outputs.hidden_states,
            outputs.attentions
        )

ix2tag = {ix: tags.int2str(ix) for ix in range(tags.num_classes)}
tag2ix = {tag: tags.str2int(tag) for tag in tags.names}

xlmr_config = AutoConfig.from_pretrained(
    xlmr_model_name,
    num_labels=tags.num_classes,
    id2label=ix2tag,
    label2id=tag2ix)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

xlmr_model = (XLMRobertaForTokenClassification
              .from_pretrained(xlmr_model_name, config=xlmr_config)
              .to(device))

IGNORE_TAG = -100

def tokenize_and_align_labels(examples: dict) -> BatchEncoding:
    tokenized_inputs = xlmr_tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True)
    labels = []
    for idx, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=idx)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                label_ids.append(IGNORE_TAG)
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

def encode_panx_dataset(corpus: Union[DatasetDict, Dataset]):
    return corpus.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=['langs', 'ner_tags', 'tokens']
    )

panx_de_encoded = encode_panx_dataset(panx_ch['de'])

def align_predictions(
    predictions: torch.Tensor,  # [batch_size, seq_len, num_tags]
    label_ids: torch.Tensor     # [batch_size, seq_len]
) -> tuple[list[list[str]], list[list[str]]]:
    '''
    Converts the predictions and label_ids tensors
    into a format that can be used by `seqeval.metrics.classification_report`, (preds, labels).
    '''
    preds: np.ndarray = np.argmax(predictions, axis=-1)
    batch_size, seq_len = preds.shape
    labels_list, preds_list = [], []
    for batch_idx in range(batch_size):
        examples_labels, example_preds = [], []
        for seq_idx in range(seq_len):
            if label_ids[batch_idx, seq_idx] != -100:
                examples_labels.append(ix2tag[label_ids[batch_idx][seq_idx]])
                example_preds.append(ix2tag[preds[batch_idx][seq_idx]])
        labels_list.append(examples_labels)
        preds_list.append(example_preds)
    return preds_list, labels_list

num_epochs = 3
batch_size = 24
logging_steps = len(panx_de_encoded['train'])// batch_size
model_name = f'{xlmr_model_name}-finetuned-panx-de'
training_args = TrainingArguments(
    output_dir=model_name,
    log_level='error',
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy='epoch',
    save_steps=1e6,
    weight_decay=0.01,
    disable_tqdm=False,
    logging_steps=logging_steps,
    push_to_hub=True
)

# we need a function to tell the Trainer how to compute metrics on the validation set
def compute_metrics(eval_pred: dict):
    y_pred, y_true = align_predictions(eval_pred.predictions, eval_pred.label_ids)
    return {'f1': f1_score(y_true, y_pred)}

data_collator = DataCollatorForTokenClassification(xlmr_tokenizer)

def model_init():
    return (XLMRobertaForTokenClassification
            .from_pretrained(xlmr_model_name, config=xlmr_config)
            .to(device))

logging.set_verbosity_info()

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    train_dataset=panx_de_encoded['train'],
    eval_dataset=panx_de_encoded['validation'],
    tokenizer=xlmr_tokenizer
)

trainer.train()
trainer.push_to_hub(commit_message='finetuned on panx-de')