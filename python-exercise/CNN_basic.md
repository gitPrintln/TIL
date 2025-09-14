## torchvision의 EMNIST, KMNIST, Fashion-MNIST(GrayScale), CIFAR10, SVHN, STL10(RGB) 데이터셋을 읽어들이고, transforms로 변환한 후 주어진 label(target)로 classify한다. 


## [Convolution animated](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)

## [nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

## [nn.MaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)


```python
import torch
import torch.nn as nn
```


```python
input = torch.randn(20, 16, 50, 32)

# output size = (input size - dialation * (kernel size - 1) - 1 + 2 * paddning) / stride + 1
# output size = (input size - kernel size + 2 * paddning) / stride + 1
# output size = (input size - kernel size) / kernel_size + 1 = (input_size / kernel_size) - 1 + 1 = (input_size / kernel_size) 

m = nn.MaxPool2d(2) # kernel size = 2, stride = 2
print(m)
output = m(input)
print(output.shape)

# pool of square window of size=3, stride=2
# (50 - 3 + 2 * 0) / 2 + 1 = 24 
# (32 - 3 + 2 * 0) / 2 + 1 = 15 
m = nn.MaxPool2d(3, stride=2)
output = m(input)
print(output.shape)

# pool of non-square window
# (50 - 3 + 2 * 0) / 2 + 1 = 24 
# (32 - 2 + 2 * 0) / 1 + 1 = 15 
m = nn.MaxPool2d((3, 2), stride=(2, 1))
output = m(input)
print(output.shape)
```

    MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    torch.Size([20, 16, 25, 16])
    torch.Size([20, 16, 24, 15])
    torch.Size([20, 16, 24, 31])


## [nn.Conv1d](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)


```python
m = nn.Conv1d(16, 33, 3, stride=2)
input = torch.randn(20, 16, 50)
output = m(input)
print(output.shape)
```

    torch.Size([20, 33, 24])


## [MaxPool1d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html)


```python
# pool of size=3, stride=2
m = nn.MaxPool1d(3, stride=2)
input = torch.randn(20, 16, 50)
output = m(input)
print(output.shape)
```

    torch.Size([20, 16, 24])

