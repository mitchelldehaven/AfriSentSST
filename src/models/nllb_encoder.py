from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaLMHead
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel


class NLLB_Encoder(pl.LightningModule):
    def __init__(self, model_type, tokenizer=None, steps_per_epoch=None, epochs=None, lr=3e-6, loss_fct_params={}, use_cls_token=False):
        super().__init__()
        self.model_type = model_type
        self.model = AutoModel.from_pretrained(self.model_type).encoder
        if tokenizer:
            self.tokenizer = tokenizer
            # self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        self.lr = lr
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.classification_layer = nn.Linear(1024, 3)
        self.loss_fct = nn.CrossEntropyLoss()
        self.use_cls_token = use_cls_token
    

    def forward(self, inputs, labels=None):
        embeddings = self.model(**inputs).last_hidden_state
        if self.use_cls_token:
            logits = self.classification_layer(embeddings[:,0])
        else:
            expanded_input_mask = inputs["attention_mask"].unsqueeze(-1).expand(embeddings.size())
            sum_embeddings = torch.sum(embeddings * expanded_input_mask, 1)
            sum_mask = torch.clamp(expanded_input_mask.sum(1), min=1e-9)
            logits = self.classification_layer(sum_embeddings / sum_mask)
        if labels is not None:
            loss = self.loss_fct(logits, labels)
        else:
            loss = None
        return logits, loss


    def training_step(self, batch):
        x, y = batch
        logits, loss = self.forward(x, y)
        y_hat = torch.argmax(logits, dim=1)
        accuracy = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.forward(x, y)
        y_hat = torch.argmax(logits, dim=1)
        accuracy = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        self.log("valid_loss", loss.detach())
        self.log("valid_accuracy", accuracy)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.lr, pct_start=0.05, steps_per_epoch=self.steps_per_epoch, epochs=self.epochs, anneal_strategy="linear")
        scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_dict]
