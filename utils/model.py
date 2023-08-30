
import torch.nn.functional as f
import torchmetrics
from torchmetrics.functional import accuracy
import pytorch_lightning as pl
from torch import Tensor
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import Precision
from torchmetrics import Recall
from torchmetrics import F1Score
from torchmetrics import ConfusionMatrix


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = f.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)

def position_encoding(
    seq_len: int, dim_model: int, device: torch.device = device,
) -> Tensor:
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** (dim // dim_model))

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input),
    )

class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )

class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src: Tensor) -> Tensor:
        src = self.attention(src, src, src)
        return self.feed_forward(src)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 2,
        dim_model: int = 49,
        num_heads: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src: Tensor) -> Tensor:
        seq_len, dimension = src.size(1), src.size(2)
        src += position_encoding(seq_len, dimension)
        for layer in self.layers:
            src = layer(src)

        return src

class wavmosLit(pl.LightningModule):
    def __init__(
        self
        ):
        super(wavmosLit, self).__init__()
        self.precision_all = Precision()
        self.precision_global = Precision(average='macro', num_classes=23)
        self.precision_weighted = Precision(average='weighted', num_classes = 23)
        self.precision_each = Precision(average='none', num_classes = 23)
        
        self.recall_all = Recall()
        self.recall_global = Recall(average='macro', num_classes=23)
        self.recall_weighted = Recall(average='weighted', num_classes = 23)
        self.recall_each = Recall(average='none', num_classes = 23)
        
        self.f1 = F1Score(num_classes=23)
        self.confmat = ConfusionMatrix(num_classes=23)
        self.valid_acc = torchmetrics.Accuracy()
        self.valid_acc_each =  torchmetrics.Accuracy(average='none', num_classes = 23)
        
        self.inner_dim = 49
        self.encoder = TransformerEncoder()
        #self.cls_emb = nn.Linear(1,49)
        self.last_layer = torch.nn.Linear(
            self.inner_dim,
            23
            )
        self._global_shared_layer = TORCH_GLOBAL_SHARED_LAYER
        self._output = None

    def forward(self, inputs):
        self._output = inputs.unsqueeze(1)
        for conv in self._global_shared_layer:
             self._output = conv(self._output)
        #self.temp = torch.ones([self._output.shape[0],1,49],device = device)
        #self.cls_token_emb = self.cls_emb(torch.tensor([50], dtype=torch.float,device = device))
        #self.temp[:,0,:] = self.cls_token_emb
        self._output = torch.cat((self._output,torch.ones([self._output.shape[0],1,49],device = device)),1)                         
        self._output = self.encoder(self._output)
        model_out = self.last_layer(self._output[:,256])
        return model_out

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        
        outputs = self(inputs)
        
        loss =  loss_function(outputs, targets)

        return {"loss": loss}

    def setup(self,stage = False):

        train_ids,test_ids = spilts[1] 
        self.training_data = CustomImageDataset(dataset)
        self.train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        self.test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    def train_dataloader(self):
    
        train_loader = torch.utils.data.DataLoader(
                      self.training_data, 
                      batch_size=64, sampler=self.train_subsampler,num_workers = 8,pin_memory=True)
        return train_loader

    def val_dataloader(self):
    
        val_loader = torch.utils.data.DataLoader(
                      self.training_data,
                      batch_size=64, sampler=self.test_subsampler,num_workers = 8,pin_memory=True)
        return val_loader
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        
        outputs = self(inputs)
        
        loss =  loss_function(outputs, targets)
        self.valid_acc.update(outputs, targets)
        self.valid_acc_each.update(outputs, targets)
        self.recall_all.update(outputs, targets)
        self.recall_global.update(outputs, targets)
        self.recall_weighted.update(outputs, targets)
        self.recall_each.update(outputs, targets)
        self.precision_all.update(outputs, targets)
        self.precision_global.update(outputs, targets)
        self.precision_weighted.update(outputs, targets)
        self.precision_each.update(outputs, targets)
        self.confmat.update(outputs, targets)
        
        return {"val_loss": loss}
    
    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        self.log("T_avg_loss", avg_loss,prog_bar = True)
        wandb.log({"T_avg_loss": avg_loss})
        
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        self.log("val_loss",avg_loss,prog_bar=True)
        self.log("val_acc",self.valid_acc.compute(),prog_bar = True)
        wandb.log({"val_loss": avg_loss,
                   'val_acc':self.valid_acc.compute(),
                  "recall_all": self.recall_all.compute(),

                  "recall_global":self.recall_global.compute(),
                  "recall_weighted":self.recall_weighted.compute(),
        
                  "precision_all": self.precision_all.compute(),
                  "precision_global":self.precision_global.compute(),
                  "precision_weighted":self.precision_weighted.compute(),
    
                  #"confusion_matrix": self.confmat.compute()
                  })
        wandb.log({f"test_acc_each/acc-{ii}": loss for ii, loss in enumerate(self.valid_acc_each.compute())})
        wandb.log({f"recall_each/recall_each-{ii}": loss for ii, loss in enumerate(self.precision_each.compute())})
        wandb.log({f"precision_each/precision_each-{ii}": loss for ii, loss in enumerate(self.precision_each.compute())})
        
        # tensorboard_logs = {{"val_loss": avg_loss,
        #            'val_acc':self.valid_acc.compute(),
        #           f"test_acc_each/acc-{ii}": loss for ii, loss in enumerate(self.valid_acc_each.compute()),
        #           "recall_all": self.recall_all.compute(),
        #           f"recall_each/recall_each-{ii}": loss for ii, loss in enumerate(self.recall_global.compute()),
        #           f"precision_each/precision_each-{ii}": loss for ii, loss in enumerate(self.precision_each.compute()),
        #           "recall_global":self.recall_global.compute(),
        #           "recall_weighted":self.recall_weighted.compute(),
        #           #"recall_each":self.recall_each.compute(),
        #           "precision_all": self.precision_all.compute(),
        #           "precision_global":self.precision_global.compute(),
        #           "precision_weighted":self.precision_weighted.compute(),
        #           #"precision_each": self.precision_each.compute(),
        #           #"confusion_matrix": self.confmat.compute()
        #           }}
        
        print(list(self.training_data.lab_dict.items()))
        print(self.confmat.compute())
        self.confmat.reset()
        self.valid_acc.reset()
        self.valid_acc_each.reset()
        self.recall_all.reset()
        self.recall_global.reset()
        self.recall_weighted.reset()
        self.recall_each.reset()
        self.precision_all.reset()
        self.precision_global.reset()
        self.precision_weighted.reset()
        self.precision_each.reset()
        return {'val_loss': avg_loss}#, 'log': tensorboard_logs}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2*(10**(-4)),capturable = True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60], gamma=0.1)
        return [optimizer], [scheduler]
