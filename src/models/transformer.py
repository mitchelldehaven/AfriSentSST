from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaLMHead
import pytorch_lightning as pl
import torch


class Transformer(pl.LightningModule):
    def __init__(self, model_type, steps_per_epoch=None, epochs=None, lr=3e-6):
        super().__init__()
        self.model_type = model_type
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_type, num_labels=3)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        self.lr = lr
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
    
    
    def forward(self, inputs, labels=None):
        outputs = self.model(**inputs, labels=labels)
        return outputs


    def training_step(self, batch):
        x, y = batch
        output = self.forward(x, y)
        y_hat = torch.argmax(output.logits, dim=1)
        loss = output.loss
        accuracy = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x, y)
        y_hat = torch.argmax(output.logits, dim=1)
        loss = output.loss
        accuracy = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        self.log("valid_loss", loss.detach())
        self.log("valid_accuracy", accuracy)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.lr, pct_start=0.05, steps_per_epoch=self.steps_per_epoch, epochs=self.epochs, anneal_strategy="linear")
        scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_dict]




class JointTransformer(pl.LightningModule):
    def __init__(self, model_type, dataloader_type, steps_per_epoch=None, epochs=None, lr=3e-6):
        super().__init__()
        self.model_type = model_type
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_type, num_labels=3)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        self.lm_head = XLMRobertaLMHead(self.model.config)
        # print(self.lm_head)
        # print("="*100)
        self.lm_head.decoder.weight = self.model.roberta.embeddings.word_embeddings.weight
        # print(self.lm_head)
        self.lr = lr
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.dataloader_type = dataloader_type
    
    
    def forward(self, inputs, labels=None, output_hidden_states=False):
        # print(inputs, flush=True)
        outputs = self.model(**inputs, labels=labels, output_hidden_states=output_hidden_states)
        return outputs


    def mlm_training_step(self, batch):
        x, y = batch
        output = self.forward(x, output_hidden_states=True)
        hidden_states = output[1][-1]
        lm_output = self.lm_head(hidden_states)
        # print(lm_output)
        # print(lm_output.shape)
        # print(y.shape)
        # print(lm_output.shape)
        loss = self.cross_entropy(lm_output.view(-1, 250002), y.view(-1))
        return loss


    def sentiment_training_step(self, batch):
        x, y = batch
        output = self.forward(x, y)
        y_hat = torch.argmax(output.logits, dim=1)
        loss = output.loss
        accuracy = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        return loss, accuracy

    
    def training_step(self, batch, batch_idx):
        if type(batch) == list:
            sentiment_loss, accuracy = self.sentiment_training_step(batch[0])
            mlm_loss = self.mlm_training_step(batch[1])
            loss = (0.05 * mlm_loss) + sentiment_loss
        elif self.dataloader_type == "sentiment":
            loss, accuracy = self.sentiment_training_step(batch)
            self.log("train_accuracy", accuracy)
        elif self.dataloader_type == "mlm":
            loss = self.mlm_training_step(batch)
        self.log("train_loss", loss)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x, y)
        y_hat = torch.argmax(output.logits, dim=1)
        loss = output.loss
        accuracy = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        self.log("valid_loss", loss.detach())
        self.log("valid_accuracy", accuracy)
        return loss
    

    def configure_optimizers(self):
        print("setting")
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.lr, pct_start=0.05, steps_per_epoch=self.steps_per_epoch, epochs=self.epochs, anneal_strategy="linear")
        scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_dict]



class MLMTransformer(pl.LightningModule):
    def __init__(self, model_type, dataloader_type, steps_per_epoch=None, epochs=None, lr=3e-6):
        super().__init__()
        self.model_type = model_type
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_type, num_labels=3)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        self.lr = lr
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.dataloader_type = dataloader_type
    
    
    def forward(self, inputs, labels=None, output_hidden_states=False):
        outputs = self.model(**inputs, labels=labels)
        return outputs


    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x, y)
        loss = output.loss
        self.log("train_loss", loss)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x, y)
        loss = output.loss
        self.log("valid_loss", loss)
        return loss
    

    def configure_optimizers(self):
        print("setting")
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.lr, pct_start=0.05, steps_per_epoch=self.steps_per_epoch, epochs=self.epochs, anneal_strategy="linear")
        scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_dict]

