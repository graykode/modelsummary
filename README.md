## modelsummary (Pytorch Model summary)

> Keras style model.summary() in PyTorch, [torchsummary](https://github.com/sksq96/pytorch-summary)

This is Pytorch library for visualization Improved tool of [torchsummary](https://github.com/sksq96/pytorch-summary) and [torchsummaryX](https://github.com/nmhkahn/torchsummaryX). I was inspired by [torchsummary](https://github.com/sksq96/pytorch-summary) and I written down code which i referred to. **It is not care with number of Input parameter!** 

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelsummary import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# show input shape
summary(Net(), torch.zeros((1, 1, 28, 28)), show_input=True)

# show output shape
summary(Net(), torch.zeros((1, 1, 28, 28)), show_input=False)
```

```
-----------------------------------------------------------------------
             Layer (type)                Input Shape         Param #
=======================================================================
                 Conv2d-1            [-1, 1, 28, 28]             260
                 Conv2d-2           [-1, 10, 12, 12]           5,020
              Dropout2d-3             [-1, 20, 8, 8]               0
                 Linear-4                  [-1, 320]          16,050
                 Linear-5                   [-1, 50]             510
=======================================================================
Total params: 21,840
Trainable params: 21,840
Non-trainable params: 0
-----------------------------------------------------------------------

-----------------------------------------------------------------------
             Layer (type)               Output Shape         Param #
=======================================================================
                 Conv2d-1           [-1, 10, 24, 24]             260
                 Conv2d-2             [-1, 20, 8, 8]           5,020
              Dropout2d-3             [-1, 20, 8, 8]               0
                 Linear-4                   [-1, 50]          16,050
                 Linear-5                   [-1, 10]             510
=======================================================================
Total params: 21,840
Trainable params: 21,840
Non-trainable params: 0
-----------------------------------------------------------------------
```



## Quick Start 

Just download with pip `modelsummary`

`pip install modelsummary` and `from modelsummary import summary`

You can use this library like this. If you see more detail, Please see example code.

```
from modelsummary import summary

model = your_model_name()

# show input shape
summary(model, (input tensor you want), show_input=True)

# show output shape
summary(model, (input tensor you want), show_input=True)

# show hierarchical struct
summary(model, (input tensor you want), hierarchical=True)
```



summary function has this parameter options`def summary(model, *inputs, batch_size=-1, show_input=True, hierarchical=False)`

#### Options

- model : your model class
- *input : your input tensor **datas** (Asterisk)
- batch_size : `-1` is same with tensor `None`
- show_input : show input shape data, **if this parameter is False, it will show output shape** **default : True**
- hierarchical : show hierarchical data structure, **default : False**



## Result

Run example using Transformer Model in [Attention is all you need paper(2017)](https://arxiv.org/abs/1706.03762)

1) showing input shape

```
# show input shape
summary(model, enc_inputs, dec_inputs, show_input=True)

-----------------------------------------------------------------------
             Layer (type)                Input Shape         Param #
=======================================================================
                Encoder-1                    [-1, 5]               0
              Embedding-2                    [-1, 5]           3,072
              Embedding-3                    [-1, 5]           3,072
           EncoderLayer-4               [-1, 5, 512]               0
     MultiHeadAttention-5               [-1, 5, 512]               0
                 Linear-6               [-1, 5, 512]         262,656
                 Linear-7               [-1, 5, 512]         262,656
                 Linear-8               [-1, 5, 512]         262,656
  PoswiseFeedForwardNet-9               [-1, 5, 512]               0
                Conv1d-10               [-1, 512, 5]       1,050,624
                Conv1d-11              [-1, 2048, 5]       1,049,088
          EncoderLayer-12               [-1, 5, 512]               0
    MultiHeadAttention-13               [-1, 5, 512]               0
                Linear-14               [-1, 5, 512]         262,656
                Linear-15               [-1, 5, 512]         262,656
                Linear-16               [-1, 5, 512]         262,656
 PoswiseFeedForwardNet-17               [-1, 5, 512]               0
                Conv1d-18               [-1, 512, 5]       1,050,624
                Conv1d-19              [-1, 2048, 5]       1,049,088
          EncoderLayer-20               [-1, 5, 512]               0
    MultiHeadAttention-21               [-1, 5, 512]               0
                Linear-22               [-1, 5, 512]         262,656
                Linear-23               [-1, 5, 512]         262,656
                Linear-24               [-1, 5, 512]         262,656
 PoswiseFeedForwardNet-25               [-1, 5, 512]               0
                Conv1d-26               [-1, 512, 5]       1,050,624
                Conv1d-27              [-1, 2048, 5]       1,049,088
          EncoderLayer-28               [-1, 5, 512]               0
    MultiHeadAttention-29               [-1, 5, 512]               0
                Linear-30               [-1, 5, 512]         262,656
                Linear-31               [-1, 5, 512]         262,656
                Linear-32               [-1, 5, 512]         262,656
 PoswiseFeedForwardNet-33               [-1, 5, 512]               0
                Conv1d-34               [-1, 512, 5]       1,050,624
                Conv1d-35              [-1, 2048, 5]       1,049,088
          EncoderLayer-36               [-1, 5, 512]               0
    MultiHeadAttention-37               [-1, 5, 512]               0
                Linear-38               [-1, 5, 512]         262,656
                Linear-39               [-1, 5, 512]         262,656
                Linear-40               [-1, 5, 512]         262,656
 PoswiseFeedForwardNet-41               [-1, 5, 512]               0
                Conv1d-42               [-1, 512, 5]       1,050,624
                Conv1d-43              [-1, 2048, 5]       1,049,088
          EncoderLayer-44               [-1, 5, 512]               0
    MultiHeadAttention-45               [-1, 5, 512]               0
                Linear-46               [-1, 5, 512]         262,656
                Linear-47               [-1, 5, 512]         262,656
                Linear-48               [-1, 5, 512]         262,656
 PoswiseFeedForwardNet-49               [-1, 5, 512]               0
                Conv1d-50               [-1, 512, 5]       1,050,624
                Conv1d-51              [-1, 2048, 5]       1,049,088
               Decoder-52                    [-1, 5]               0
             Embedding-53                    [-1, 5]           3,584
             Embedding-54                    [-1, 5]           3,072
          DecoderLayer-55               [-1, 5, 512]               0
    MultiHeadAttention-56               [-1, 5, 512]               0
                Linear-57               [-1, 5, 512]         262,656
                Linear-58               [-1, 5, 512]         262,656
                Linear-59               [-1, 5, 512]         262,656
    MultiHeadAttention-60               [-1, 5, 512]               0
                Linear-61               [-1, 5, 512]         262,656
                Linear-62               [-1, 5, 512]         262,656
                Linear-63               [-1, 5, 512]         262,656
 PoswiseFeedForwardNet-64               [-1, 5, 512]               0
                Conv1d-65               [-1, 512, 5]       1,050,624
                Conv1d-66              [-1, 2048, 5]       1,049,088
          DecoderLayer-67               [-1, 5, 512]               0
    MultiHeadAttention-68               [-1, 5, 512]               0
                Linear-69               [-1, 5, 512]         262,656
                Linear-70               [-1, 5, 512]         262,656
                Linear-71               [-1, 5, 512]         262,656
    MultiHeadAttention-72               [-1, 5, 512]               0
                Linear-73               [-1, 5, 512]         262,656
                Linear-74               [-1, 5, 512]         262,656
                Linear-75               [-1, 5, 512]         262,656
 PoswiseFeedForwardNet-76               [-1, 5, 512]               0
                Conv1d-77               [-1, 512, 5]       1,050,624
                Conv1d-78              [-1, 2048, 5]       1,049,088
          DecoderLayer-79               [-1, 5, 512]               0
    MultiHeadAttention-80               [-1, 5, 512]               0
                Linear-81               [-1, 5, 512]         262,656
                Linear-82               [-1, 5, 512]         262,656
                Linear-83               [-1, 5, 512]         262,656
    MultiHeadAttention-84               [-1, 5, 512]               0
                Linear-85               [-1, 5, 512]         262,656
                Linear-86               [-1, 5, 512]         262,656
                Linear-87               [-1, 5, 512]         262,656
 PoswiseFeedForwardNet-88               [-1, 5, 512]               0
                Conv1d-89               [-1, 512, 5]       1,050,624
                Conv1d-90              [-1, 2048, 5]       1,049,088
          DecoderLayer-91               [-1, 5, 512]               0
    MultiHeadAttention-92               [-1, 5, 512]               0
                Linear-93               [-1, 5, 512]         262,656
                Linear-94               [-1, 5, 512]         262,656
                Linear-95               [-1, 5, 512]         262,656
    MultiHeadAttention-96               [-1, 5, 512]               0
                Linear-97               [-1, 5, 512]         262,656
                Linear-98               [-1, 5, 512]         262,656
                Linear-99               [-1, 5, 512]         262,656
PoswiseFeedForwardNet-100               [-1, 5, 512]               0
               Conv1d-101               [-1, 512, 5]       1,050,624
               Conv1d-102              [-1, 2048, 5]       1,049,088
         DecoderLayer-103               [-1, 5, 512]               0
   MultiHeadAttention-104               [-1, 5, 512]               0
               Linear-105               [-1, 5, 512]         262,656
               Linear-106               [-1, 5, 512]         262,656
               Linear-107               [-1, 5, 512]         262,656
   MultiHeadAttention-108               [-1, 5, 512]               0
               Linear-109               [-1, 5, 512]         262,656
               Linear-110               [-1, 5, 512]         262,656
               Linear-111               [-1, 5, 512]         262,656
PoswiseFeedForwardNet-112               [-1, 5, 512]               0
               Conv1d-113               [-1, 512, 5]       1,050,624
               Conv1d-114              [-1, 2048, 5]       1,049,088
         DecoderLayer-115               [-1, 5, 512]               0
   MultiHeadAttention-116               [-1, 5, 512]               0
               Linear-117               [-1, 5, 512]         262,656
               Linear-118               [-1, 5, 512]         262,656
               Linear-119               [-1, 5, 512]         262,656
   MultiHeadAttention-120               [-1, 5, 512]               0
               Linear-121               [-1, 5, 512]         262,656
               Linear-122               [-1, 5, 512]         262,656
               Linear-123               [-1, 5, 512]         262,656
PoswiseFeedForwardNet-124               [-1, 5, 512]               0
               Conv1d-125               [-1, 512, 5]       1,050,624
               Conv1d-126              [-1, 2048, 5]       1,049,088
               Linear-127               [-1, 5, 512]           3,584
=======================================================================
Total params: 39,396,352
Trainable params: 39,390,208
Non-trainable params: 6,144
```

2) showing output shape

```
# show output shape
summary(model, enc_inputs, dec_inputs, show_input=False)

-----------------------------------------------------------------------
             Layer (type)               Output Shape         Param #
=======================================================================
              Embedding-1               [-1, 5, 512]           3,072
              Embedding-2               [-1, 5, 512]           3,072
                 Linear-3               [-1, 5, 512]         262,656
                 Linear-4               [-1, 5, 512]         262,656
                 Linear-5               [-1, 5, 512]         262,656
     MultiHeadAttention-6              [-1, 8, 5, 5]               0
                 Conv1d-7              [-1, 2048, 5]       1,050,624
                 Conv1d-8               [-1, 512, 5]       1,049,088
  PoswiseFeedForwardNet-9               [-1, 5, 512]               0
          EncoderLayer-10              [-1, 8, 5, 5]               0
                Linear-11               [-1, 5, 512]         262,656
                Linear-12               [-1, 5, 512]         262,656
                Linear-13               [-1, 5, 512]         262,656
    MultiHeadAttention-14              [-1, 8, 5, 5]               0
                Conv1d-15              [-1, 2048, 5]       1,050,624
                Conv1d-16               [-1, 512, 5]       1,049,088
 PoswiseFeedForwardNet-17               [-1, 5, 512]               0
          EncoderLayer-18              [-1, 8, 5, 5]               0
                Linear-19               [-1, 5, 512]         262,656
                Linear-20               [-1, 5, 512]         262,656
                Linear-21               [-1, 5, 512]         262,656
    MultiHeadAttention-22              [-1, 8, 5, 5]               0
                Conv1d-23              [-1, 2048, 5]       1,050,624
                Conv1d-24               [-1, 512, 5]       1,049,088
 PoswiseFeedForwardNet-25               [-1, 5, 512]               0
          EncoderLayer-26              [-1, 8, 5, 5]               0
                Linear-27               [-1, 5, 512]         262,656
                Linear-28               [-1, 5, 512]         262,656
                Linear-29               [-1, 5, 512]         262,656
    MultiHeadAttention-30              [-1, 8, 5, 5]               0
                Conv1d-31              [-1, 2048, 5]       1,050,624
                Conv1d-32               [-1, 512, 5]       1,049,088
 PoswiseFeedForwardNet-33               [-1, 5, 512]               0
          EncoderLayer-34              [-1, 8, 5, 5]               0
                Linear-35               [-1, 5, 512]         262,656
                Linear-36               [-1, 5, 512]         262,656
                Linear-37               [-1, 5, 512]         262,656
    MultiHeadAttention-38              [-1, 8, 5, 5]               0
                Conv1d-39              [-1, 2048, 5]       1,050,624
                Conv1d-40               [-1, 512, 5]       1,049,088
 PoswiseFeedForwardNet-41               [-1, 5, 512]               0
          EncoderLayer-42              [-1, 8, 5, 5]               0
                Linear-43               [-1, 5, 512]         262,656
                Linear-44               [-1, 5, 512]         262,656
                Linear-45               [-1, 5, 512]         262,656
    MultiHeadAttention-46              [-1, 8, 5, 5]               0
                Conv1d-47              [-1, 2048, 5]       1,050,624
                Conv1d-48               [-1, 512, 5]       1,049,088
 PoswiseFeedForwardNet-49               [-1, 5, 512]               0
          EncoderLayer-50              [-1, 8, 5, 5]               0
               Encoder-51            [[-1, 8, 5, 5]]               0
             Embedding-52               [-1, 5, 512]           3,584
             Embedding-53               [-1, 5, 512]           3,072
                Linear-54               [-1, 5, 512]         262,656
                Linear-55               [-1, 5, 512]         262,656
                Linear-56               [-1, 5, 512]         262,656
    MultiHeadAttention-57              [-1, 8, 5, 5]               0
                Linear-58               [-1, 5, 512]         262,656
                Linear-59               [-1, 5, 512]         262,656
                Linear-60               [-1, 5, 512]         262,656
    MultiHeadAttention-61              [-1, 8, 5, 5]               0
                Conv1d-62              [-1, 2048, 5]       1,050,624
                Conv1d-63               [-1, 512, 5]       1,049,088
 PoswiseFeedForwardNet-64               [-1, 5, 512]               0
          DecoderLayer-65              [-1, 8, 5, 5]               0
                Linear-66               [-1, 5, 512]         262,656
                Linear-67               [-1, 5, 512]         262,656
                Linear-68               [-1, 5, 512]         262,656
    MultiHeadAttention-69              [-1, 8, 5, 5]               0
                Linear-70               [-1, 5, 512]         262,656
                Linear-71               [-1, 5, 512]         262,656
                Linear-72               [-1, 5, 512]         262,656
    MultiHeadAttention-73              [-1, 8, 5, 5]               0
                Conv1d-74              [-1, 2048, 5]       1,050,624
                Conv1d-75               [-1, 512, 5]       1,049,088
 PoswiseFeedForwardNet-76               [-1, 5, 512]               0
          DecoderLayer-77              [-1, 8, 5, 5]               0
                Linear-78               [-1, 5, 512]         262,656
                Linear-79               [-1, 5, 512]         262,656
                Linear-80               [-1, 5, 512]         262,656
    MultiHeadAttention-81              [-1, 8, 5, 5]               0
                Linear-82               [-1, 5, 512]         262,656
                Linear-83               [-1, 5, 512]         262,656
                Linear-84               [-1, 5, 512]         262,656
    MultiHeadAttention-85              [-1, 8, 5, 5]               0
                Conv1d-86              [-1, 2048, 5]       1,050,624
                Conv1d-87               [-1, 512, 5]       1,049,088
 PoswiseFeedForwardNet-88               [-1, 5, 512]               0
          DecoderLayer-89              [-1, 8, 5, 5]               0
                Linear-90               [-1, 5, 512]         262,656
                Linear-91               [-1, 5, 512]         262,656
                Linear-92               [-1, 5, 512]         262,656
    MultiHeadAttention-93              [-1, 8, 5, 5]               0
                Linear-94               [-1, 5, 512]         262,656
                Linear-95               [-1, 5, 512]         262,656
                Linear-96               [-1, 5, 512]         262,656
    MultiHeadAttention-97              [-1, 8, 5, 5]               0
                Conv1d-98              [-1, 2048, 5]       1,050,624
                Conv1d-99               [-1, 512, 5]       1,049,088
PoswiseFeedForwardNet-100               [-1, 5, 512]               0
         DecoderLayer-101              [-1, 8, 5, 5]               0
               Linear-102               [-1, 5, 512]         262,656
               Linear-103               [-1, 5, 512]         262,656
               Linear-104               [-1, 5, 512]         262,656
   MultiHeadAttention-105              [-1, 8, 5, 5]               0
               Linear-106               [-1, 5, 512]         262,656
               Linear-107               [-1, 5, 512]         262,656
               Linear-108               [-1, 5, 512]         262,656
   MultiHeadAttention-109              [-1, 8, 5, 5]               0
               Conv1d-110              [-1, 2048, 5]       1,050,624
               Conv1d-111               [-1, 512, 5]       1,049,088
PoswiseFeedForwardNet-112               [-1, 5, 512]               0
         DecoderLayer-113              [-1, 8, 5, 5]               0
               Linear-114               [-1, 5, 512]         262,656
               Linear-115               [-1, 5, 512]         262,656
               Linear-116               [-1, 5, 512]         262,656
   MultiHeadAttention-117              [-1, 8, 5, 5]               0
               Linear-118               [-1, 5, 512]         262,656
               Linear-119               [-1, 5, 512]         262,656
               Linear-120               [-1, 5, 512]         262,656
   MultiHeadAttention-121              [-1, 8, 5, 5]               0
               Conv1d-122              [-1, 2048, 5]       1,050,624
               Conv1d-123               [-1, 512, 5]       1,049,088
PoswiseFeedForwardNet-124               [-1, 5, 512]               0
         DecoderLayer-125              [-1, 8, 5, 5]               0
              Decoder-126            [[-1, 8, 5, 5]]               0
               Linear-127                 [-1, 5, 7]           3,584
=======================================================================
Total params: 39,396,352
Trainable params: 39,390,208
Non-trainable params: 6,144
-----------------------------------------------------------------------
```

3) showing hierarchical summary

```
Transformer(
  (encoder): Encoder(
    (src_emb): Embedding(6, 512), 3,072 params
    (pos_emb): Embedding(6, 512), 3,072 params
    (layers): ModuleList(
      (0): EncoderLayer(
        (enc_self_attn): MultiHeadAttention(
          (W_Q): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_K): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_V): Linear(in_features=512, out_features=512, bias=True), 262,656 params
        ), 787,968 params
        (pos_ffn): PoswiseFeedForwardNet(
          (conv1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,)), 1,050,624 params
          (conv2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,)), 1,049,088 params
        ), 2,099,712 params
      ), 2,887,680 params
      (1): EncoderLayer(
        (enc_self_attn): MultiHeadAttention(
          (W_Q): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_K): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_V): Linear(in_features=512, out_features=512, bias=True), 262,656 params
        ), 787,968 params
        (pos_ffn): PoswiseFeedForwardNet(
          (conv1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,)), 1,050,624 params
          (conv2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,)), 1,049,088 params
        ), 2,099,712 params
      ), 2,887,680 params
      (2): EncoderLayer(
        (enc_self_attn): MultiHeadAttention(
          (W_Q): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_K): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_V): Linear(in_features=512, out_features=512, bias=True), 262,656 params
        ), 787,968 params
        (pos_ffn): PoswiseFeedForwardNet(
          (conv1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,)), 1,050,624 params
          (conv2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,)), 1,049,088 params
        ), 2,099,712 params
      ), 2,887,680 params
      (3): EncoderLayer(
        (enc_self_attn): MultiHeadAttention(
          (W_Q): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_K): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_V): Linear(in_features=512, out_features=512, bias=True), 262,656 params
        ), 787,968 params
        (pos_ffn): PoswiseFeedForwardNet(
          (conv1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,)), 1,050,624 params
          (conv2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,)), 1,049,088 params
        ), 2,099,712 params
      ), 2,887,680 params
      (4): EncoderLayer(
        (enc_self_attn): MultiHeadAttention(
          (W_Q): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_K): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_V): Linear(in_features=512, out_features=512, bias=True), 262,656 params
        ), 787,968 params
        (pos_ffn): PoswiseFeedForwardNet(
          (conv1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,)), 1,050,624 params
          (conv2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,)), 1,049,088 params
        ), 2,099,712 params
      ), 2,887,680 params
      (5): EncoderLayer(
        (enc_self_attn): MultiHeadAttention(
          (W_Q): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_K): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_V): Linear(in_features=512, out_features=512, bias=True), 262,656 params
        ), 787,968 params
        (pos_ffn): PoswiseFeedForwardNet(
          (conv1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,)), 1,050,624 params
          (conv2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,)), 1,049,088 params
        ), 2,099,712 params
      ), 2,887,680 params
    ), 17,326,080 params
  ), 17,332,224 params
  (decoder): Decoder(
    (tgt_emb): Embedding(7, 512), 3,584 params
    (pos_emb): Embedding(6, 512), 3,072 params
    (layers): ModuleList(
      (0): DecoderLayer(
        (dec_self_attn): MultiHeadAttention(
          (W_Q): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_K): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_V): Linear(in_features=512, out_features=512, bias=True), 262,656 params
        ), 787,968 params
        (dec_enc_attn): MultiHeadAttention(
          (W_Q): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_K): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_V): Linear(in_features=512, out_features=512, bias=True), 262,656 params
        ), 787,968 params
        (pos_ffn): PoswiseFeedForwardNet(
          (conv1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,)), 1,050,624 params
          (conv2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,)), 1,049,088 params
        ), 2,099,712 params
      ), 3,675,648 params
      (1): DecoderLayer(
        (dec_self_attn): MultiHeadAttention(
          (W_Q): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_K): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_V): Linear(in_features=512, out_features=512, bias=True), 262,656 params
        ), 787,968 params
        (dec_enc_attn): MultiHeadAttention(
          (W_Q): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_K): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_V): Linear(in_features=512, out_features=512, bias=True), 262,656 params
        ), 787,968 params
        (pos_ffn): PoswiseFeedForwardNet(
          (conv1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,)), 1,050,624 params
          (conv2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,)), 1,049,088 params
        ), 2,099,712 params
      ), 3,675,648 params
      (2): DecoderLayer(
        (dec_self_attn): MultiHeadAttention(
          (W_Q): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_K): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_V): Linear(in_features=512, out_features=512, bias=True), 262,656 params
        ), 787,968 params
        (dec_enc_attn): MultiHeadAttention(
          (W_Q): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_K): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_V): Linear(in_features=512, out_features=512, bias=True), 262,656 params
        ), 787,968 params
        (pos_ffn): PoswiseFeedForwardNet(
          (conv1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,)), 1,050,624 params
          (conv2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,)), 1,049,088 params
        ), 2,099,712 params
      ), 3,675,648 params
      (3): DecoderLayer(
        (dec_self_attn): MultiHeadAttention(
          (W_Q): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_K): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_V): Linear(in_features=512, out_features=512, bias=True), 262,656 params
        ), 787,968 params
        (dec_enc_attn): MultiHeadAttention(
          (W_Q): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_K): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_V): Linear(in_features=512, out_features=512, bias=True), 262,656 params
        ), 787,968 params
        (pos_ffn): PoswiseFeedForwardNet(
          (conv1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,)), 1,050,624 params
          (conv2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,)), 1,049,088 params
        ), 2,099,712 params
      ), 3,675,648 params
      (4): DecoderLayer(
        (dec_self_attn): MultiHeadAttention(
          (W_Q): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_K): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_V): Linear(in_features=512, out_features=512, bias=True), 262,656 params
        ), 787,968 params
        (dec_enc_attn): MultiHeadAttention(
          (W_Q): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_K): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_V): Linear(in_features=512, out_features=512, bias=True), 262,656 params
        ), 787,968 params
        (pos_ffn): PoswiseFeedForwardNet(
          (conv1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,)), 1,050,624 params
          (conv2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,)), 1,049,088 params
        ), 2,099,712 params
      ), 3,675,648 params
      (5): DecoderLayer(
        (dec_self_attn): MultiHeadAttention(
          (W_Q): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_K): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_V): Linear(in_features=512, out_features=512, bias=True), 262,656 params
        ), 787,968 params
        (dec_enc_attn): MultiHeadAttention(
          (W_Q): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_K): Linear(in_features=512, out_features=512, bias=True), 262,656 params
          (W_V): Linear(in_features=512, out_features=512, bias=True), 262,656 params
        ), 787,968 params
        (pos_ffn): PoswiseFeedForwardNet(
          (conv1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,)), 1,050,624 params
          (conv2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,)), 1,049,088 params
        ), 2,099,712 params
      ), 3,675,648 params
    ), 22,053,888 params
  ), 22,060,544 params
  (projection): Linear(in_features=512, out_features=7, bias=False), 3,584 params
), 39,396,352 params

```



## Reference

```python
code_reference = { 'https://github.com/pytorch/pytorch/issues/2001',
				'https://gist.github.com/HTLife/b6640af9d6e7d765411f8aa9aa94b837',
				'https://github.com/sksq96/pytorch-summary',
				'Inspired by https://github.com/sksq96/pytorch-summary'}
```