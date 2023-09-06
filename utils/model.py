import torch
import torch.nn as nn
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
from config import stopping_threshold


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class TransposeLast(nn.Module):
    """
    A module that transposes the last two dimensions of a tensor.

    Args:
        deconstruct_idx (Optional): An optional index for deconstructing the tensor.

    Example:
        transposer = TransposeLast()
        x = torch.randn(3, 4, 5)  # Input tensor with shape (3, 4, 5)
        y = transposer(x)  # Transposes the last two dimensions, resulting in shape (3, 5, 4)
    """
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)

class Fp32LayerNorm(nn.LayerNorm):
    """
    A custom LayerNorm module that applies layer normalization to input tensors.

    Args:
        *args: Variable length positional arguments.
        **kwargs: Variable length keyword arguments.

    Example:
        layer_norm = Fp32LayerNorm(256)  # Layer normalization for tensors with 256 features
        x = torch.randn(32, 256, 10)  # Input tensor
        y = layer_norm(x)  # Applies layer normalization to 'x'
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)

def conv_block(n_in, n_out, k, stride, conv_bias):
  """
    Defines a convolutional block with optional dropout and layer normalization.

    Args:
        n_in (int): Number of input channels.
        n_out (int): Number of output channels.
        k (int): Kernel size for convolution.
        stride (int): Stride for convolution.
        conv_bias (bool): Whether to include bias in convolution.

    Returns:
        nn.Sequential: A sequential container of convolutional layers with dropout and layer normalization.

    Example:
        conv = conv_block(256, 512, 3, 1, conv_bias=True)  # Create a convolutional block.
  """
  def spin_conv():
    conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
    nn.init.kaiming_normal_(conv.weight)
    return conv
  return nn.Sequential(
                    spin_conv(),
                    nn.Dropout(p=0.1),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(n_out, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
def make_conv_layers():
        """
        Creates a sequence of convolutional layers with specified dimensions and parameters.
    
        Returns:
            nn.ModuleList: A list of convolutional layers.
    
        Example:
            conv_layers = make_conv_layers()  # Create convolutional layers according to specified dimensions.
        """
        in_d = 1
        conv_dim = [(256, 10, 5)] + [(256, 3, 2)]*4 + [(256,2,2)] + [(256,2,2)]
        conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_dim):
            (dim, k, stride) = cl
            conv_layers.append(
                conv_block(
                    in_d,
                    dim,
                    k,
                    stride,
                    conv_bias=False,
                )
            )
            in_d = dim 
        return conv_layers


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    """
    Computes scaled dot-product attention.

    Args:
        query (Tensor): Query tensor.
        key (Tensor): Key tensor.
        value (Tensor): Value tensor.

    Returns:
        Tensor: Result of scaled dot-product attention.

    Example:
        attention_result = scaled_dot_product_attention(query, key, value)  # Calculate attention result.
    """
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = f.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)

def position_encoding(
    seq_len: int, dim_model: int, device: torch.device = device,
) -> Tensor:
    """
    Computes positional encoding for transformer models.

    Args:
        seq_len (int): Length of the sequence.
        dim_model (int): Dimension of the model.
        device (torch.device): Device to perform computations (default: global 'device').

    Returns:
        Tensor: Positional encoding tensor.

    Example:
        pos_encoding = position_encoding(seq_len=10, dim_model=512)  # Generate positional encoding.
    """
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** (dim // dim_model))

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    """
    Defines a feed-forward neural network layer.

    Args:
        dim_input (int): Input dimension.
        dim_feedforward (int): Dimension of the feed-forward layer.

    Returns:
        nn.Module: Feed-forward neural network.

    Example:
        ffnn = feed_forward(dim_input=512, dim_feedforward=2048)  # Create a feed-forward layer.
    """
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input),
    )

class AttentionHead(nn.Module):
    """
    A single attention head for multi-head attention.

    Args:
        dim_in (int): Input dimension.
        dim_q (int): Dimension of query vectors.
        dim_k (int): Dimension of key vectors.

    Example:
        attention_head = AttentionHead(dim_in=512, dim_q=64, dim_k=64)  # Create an attention head.
    """
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.

    Args:
        num_heads (int): Number of attention heads.
        dim_in (int): Input dimension.
        dim_q (int): Dimension of query vectors.
        dim_k (int): Dimension of key vectors.

    Example:
        multi_head_attention = MultiHeadAttention(num_heads=8, dim_in=512, dim_q=64, dim_k=64)  # Create a multi-head attention module.
    """
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
    """
    Residual connection with layer normalization and dropout.

    Args:
        sublayer (nn.Module): Sublayer to apply residual connection.
        dimension (int): Dimension of the sublayer's output.
        dropout (float): Dropout rate.

    Example:
        residual_layer = Residual(sublayer=feed_forward(), dimension=512, dropout=0.1)  # Create a residual layer.
    """
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
    """
    A single layer of the Transformer encoder.

    Args:
        dim_model (int): Model's dimension.
        num_heads (int): Number of attention heads.
        dim_feedforward (int): Dimension of the feed-forward layer.
        dropout (float): Dropout rate.

    Example:
        encoder_layer = TransformerEncoderLayer(dim_model=512, num_heads=8, dim_feedforward=2048, dropout=0.1)  # Create a Transformer encoder layer.
    """
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

early_stopping = EarlyStopping('acc',patience=5,mode='max',stopping_threshold = stopping_threshold)
loss_function = nn.CrossEntropyLoss()

class TransformerEncoder(nn.Module):
     """
    Transformer Encoder model.

    Args:
        num_layers (int): Number of encoder layers.
        dim_model (int): Model's dimension.
        num_heads (int): Number of attention heads.
        dim_feedforward (int): Dimension of the feed-forward layer.
        dropout (float): Dropout rate.

    Example:
        encoder = TransformerEncoder(num_layers=6, dim_model=512, num_heads=8, dim_feedforward=2048, dropout=0.1)  # Create a Transformer encoder model.
    """
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

TORCH_GLOBAL_SHARED_LAYER = make_conv_layers()


class wavmosLit(pl.LightningModule):
    """
        Initialize the wavmosLit LightningModule.

        This module defines a neural network for a classification task and includes
        various metrics for evaluation during training and validation.

        Attributes:
            precision_all: Precision metric for all classes.
            precision_global: Macro-averaged Precision metric.
            precision_weighted: Weighted Precision metric.
            precision_each: Precision metric for each class.

            recall_all: Recall metric for all classes.
            recall_global: Macro-averaged Recall metric.
            recall_weighted: Weighted Recall metric.
            recall_each: Recall metric for each class.

            f1: F1 Score metric for all classes.
            confmat: Confusion Matrix metric for all classes.
            valid_acc: Accuracy metric for all classes.
            valid_acc_each: Accuracy metric for each class.

            inner_dim: Dimension of the inner layers in the network.
            encoder: Transformer-based encoder network.
            last_layer: Final classification layer.
    """
    
    def __init__(
        self
        ):
        super(wavmosLit, self).__init__()
        self.precision_all = Precision(task="multiclass",num_classes=23)
        self.precision_global = Precision(task="multiclass",average='macro', num_classes=23)
        self.precision_weighted = Precision(task="multiclass",average='weighted', num_classes = 23)
        self.precision_each = Precision(task="multiclass",average='none', num_classes = 23)
        
        self.recall_all = Recall(task="multiclass",num_classes=23)
        self.recall_global = Recall(task="multiclass",average='macro', num_classes=23)
        self.recall_weighted = Recall(task="multiclass",average='weighted', num_classes = 23)
        self.recall_each = Recall(task="multiclass",average='none', num_classes = 23)
        
        self.f1 = F1Score(task="multiclass",num_classes=23)
        self.confmat = ConfusionMatrix(task="multiclass",num_classes=23)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass",num_classes=23)
        self.valid_acc_each =  torchmetrics.Accuracy(task="multiclass",average='none', num_classes = 23)
        
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
        """
        Forward pass of the model.

        Args:
            inputs: Input data of shape (batch_size, input_dim).

        Returns:
            model_out: Predicted class probabilities of shape (batch_size, num_classes).
        """
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
        """
        Training step.

        Args:
            batch: Batch of training data.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary containing the loss.
        """
        inputs, targets = batch
        
        outputs = self(inputs)
        
        loss =  loss_function(outputs, targets)

        return {"loss": loss}

    def setup(self,stage = False):
        """
        Setup method to define the training and testing data.

        Args:
            stage: Indicates if setup is for training or testing.
        """

        train_ids,test_ids = spilts[1] 
        self.training_data = CustomImageDataset(dataset)
        self.train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        self.test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    def train_dataloader(self):
        """
        Training DataLoader definition.

        Returns:
            DataLoader for training data.
        """
        
        train_loader = torch.utils.data.DataLoader(
                      self.training_data, 
                      batch_size=64, sampler=self.train_subsampler,num_workers = 8,pin_memory=True)
        return train_loader

    def val_dataloader(self):
        """
        Validation DataLoader definition.

        Returns:
            DataLoader for validation data.
        """
        val_loader = torch.utils.data.DataLoader(
                      self.training_data,
                      batch_size=64, sampler=self.test_subsampler,num_workers = 8,pin_memory=True)
        return val_loader
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch: Batch of validation data.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary containing validation loss.
        """
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
        """
        Training epoch end.

        Args:
            training_step_outputs: Outputs from all training steps in the epoch.
        """
        avg_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        self.log("T_avg_loss", avg_loss,prog_bar = True)
        wandb.log({"T_avg_loss": avg_loss})
        
    def validation_epoch_end(self, outputs):
        """
        Validation epoch end.

        Args:
            outputs: Outputs from all validation steps in the epoch.
        """
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


if __name__ == "__main__":
    model = wavmosLit()
    print(model)
    print("Done")
