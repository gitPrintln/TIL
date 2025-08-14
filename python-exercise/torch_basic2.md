## torch

## max / argmax


```python
!pip3 install torch numpy matplotlib
```

    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: torch in /home/ubuntu/.local/lib/python3.10/site-packages (2.7.1)
    Requirement already satisfied: numpy in /home/ubuntu/.local/lib/python3.10/site-packages (2.2.6)
    Requirement already satisfied: matplotlib in /home/ubuntu/.local/lib/python3.10/site-packages (3.10.5)
    Requirement already satisfied: nvidia-cusparselt-cu12==0.6.3 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch) (0.6.3)
    Requirement already satisfied: nvidia-cufft-cu12==11.3.0.4 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch) (11.3.0.4)
    Requirement already satisfied: nvidia-cuda-cupti-cu12==12.6.80 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch) (12.6.80)
    Requirement already satisfied: nvidia-cusolver-cu12==11.7.1.2 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch) (11.7.1.2)
    Requirement already satisfied: nvidia-cufile-cu12==1.11.1.6 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch) (1.11.1.6)
    Requirement already satisfied: nvidia-cudnn-cu12==9.5.1.17 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch) (9.5.1.17)
    Requirement already satisfied: nvidia-nvjitlink-cu12==12.6.85 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch) (12.6.85)
    Requirement already satisfied: nvidia-nvtx-cu12==12.6.77 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch) (12.6.77)
    Requirement already satisfied: fsspec in /home/ubuntu/.local/lib/python3.10/site-packages (from torch) (2025.7.0)
    Requirement already satisfied: triton==3.3.1 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch) (3.3.1)
    Requirement already satisfied: typing-extensions>=4.10.0 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch) (4.14.1)
    Requirement already satisfied: nvidia-cublas-cu12==12.6.4.1 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch) (12.6.4.1)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.6.77 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch) (12.6.77)
    Requirement already satisfied: nvidia-cusparse-cu12==12.5.4.2 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch) (12.5.4.2)
    Requirement already satisfied: nvidia-curand-cu12==10.3.7.77 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch) (10.3.7.77)
    Requirement already satisfied: networkx in /home/ubuntu/.local/lib/python3.10/site-packages (from torch) (3.4.2)
    Requirement already satisfied: nvidia-cuda-runtime-cu12==12.6.77 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch) (12.6.77)
    Requirement already satisfied: sympy>=1.13.3 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch) (1.14.0)
    Requirement already satisfied: jinja2 in /usr/lib/python3/dist-packages (from torch) (3.0.3)
    Requirement already satisfied: filelock in /home/ubuntu/.local/lib/python3.10/site-packages (from torch) (3.18.0)
    Requirement already satisfied: nvidia-nccl-cu12==2.26.2 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch) (2.26.2)
    Requirement already satisfied: setuptools>=40.8.0 in /usr/lib/python3/dist-packages (from triton==3.3.1->torch) (59.6.0)
    Requirement already satisfied: packaging>=20.0 in /home/ubuntu/.local/lib/python3.10/site-packages (from matplotlib) (25.0)
    Requirement already satisfied: fonttools>=4.22.0 in /home/ubuntu/.local/lib/python3.10/site-packages (from matplotlib) (4.59.0)
    Requirement already satisfied: contourpy>=1.0.1 in /home/ubuntu/.local/lib/python3.10/site-packages (from matplotlib) (1.3.2)
    Requirement already satisfied: kiwisolver>=1.3.1 in /home/ubuntu/.local/lib/python3.10/site-packages (from matplotlib) (1.4.8)
    Requirement already satisfied: pillow>=8 in /home/ubuntu/.local/lib/python3.10/site-packages (from matplotlib) (11.3.0)
    Requirement already satisfied: cycler>=0.10 in /home/ubuntu/.local/lib/python3.10/site-packages (from matplotlib) (0.12.1)
    Requirement already satisfied: python-dateutil>=2.7 in /home/ubuntu/.local/lib/python3.10/site-packages (from matplotlib) (2.9.0.post0)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/lib/python3/dist-packages (from matplotlib) (2.4.7)
    Requirement already satisfied: six>=1.5 in /usr/lib/python3/dist-packages (from python-dateutil>=2.7->matplotlib) (1.16.0)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /home/ubuntu/.local/lib/python3.10/site-packages (from sympy>=1.13.3->torch) (1.3.0)



```python
import torch

```


```python
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
print(t.shape)
```

    tensor([[1., 2.],
            [3., 4.]])
    torch.Size([2, 2])



```python
print(t.max()) # Returns one value: max
print(t.max(dim=0)) # Returns two values: max and argmax
```

    tensor(4.)
    torch.return_types.max(
    values=tensor([3., 4.]),
    indices=tensor([1, 1]))



```python
print('Max: ', t.max(dim=0)[0])
print('Argmax: ', t.max(dim=0)[1])
print('Argmax: ', t.argmax(dim=0))
```

    Max:  tensor([3., 4.])
    Argmax:  tensor([1, 1])
    Argmax:  tensor([1, 1])



```python
print(t.max(dim=1))
print(t.max(dim=-1))
```

    torch.return_types.max(
    values=tensor([2., 4.]),
    indices=tensor([1, 1]))
    torch.return_types.max(
    values=tensor([2., 4.]),
    indices=tensor([1, 1]))


## linear regression


```python
import torch

def linear(x, W, b):
    y = torch.mm(x, W) + b # mm: [n, m] x [m,p] = [n,p]
    return y

x = torch.randint(0,10,size=(3,2))
W = torch.randint(0,10,size=(2,4))
b = torch.randint(0,10,size=(4,))

y = linear(x, W, b) # [3, 2] X [2, 4] = [3, 4] + [4] = [3, 4] + [3, 4] = [3, 4]

print(x.shape)
print(W.shape)
print(b.shape)
print(y.shape)
```

    torch.Size([3, 2])
    torch.Size([2, 4])
    torch.Size([4])
    torch.Size([3, 4])



```python
print(torch.mm(x, W))
print(b)
print(y)
```

    tensor([[32, 40, 88, 56],
            [16, 20, 58, 30],
            [20, 25, 27, 31]])
    tensor([9, 2, 7, 1])
    tensor([[41, 42, 95, 57],
            [25, 22, 65, 31],
            [29, 27, 34, 32]])


## basic linear


```python
import torch
import torch.nn as nn

class MyLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.W = torch.FloatTensor(input_size, output_size)
        self.b = torch.FloatTensor(output_size)
    
    def forward(self, x):
        y = torch.mm(x, self.W) + self.b
        
        return y
    
x = torch.FloatTensor(16, 10)
linear = MyLinear(10, 5)
y = linear(x)

params =[p.size() for p in linear.parameters()]
print(params)
```

    []



```python
import torch
import torch.nn as nn

class MyLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_size, output_size), requires_grad=True)
        self.b = nn.Parameter(torch.FloatTensor(output_size), requires_grad=True)
    
    def forward(self, x):
        y = torch.mm(x, self.W) + self.b
        
        return y
    
x = torch.FloatTensor(16, 10)
linear = MyLinear(10, 5)
y = linear(x)

params =[p.size() for p in linear.parameters()]
print(params) # w, b를 추적. y로부터 미분한 기울기를 구할 수 있다
```

    [torch.Size([10, 5]), torch.Size([5])]



```python
import torch
import torch.nn as nn

class MyLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyLinear, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        y = self.linear(x)
        
        return y
    
x = torch.FloatTensor(16, 10)
linear = MyLinear(10, 5)
y = linear(x)

params =[p.size() for p in linear.parameters()]
print(params)
print(linear)
```

    [torch.Size([5, 10]), torch.Size([5])]
    MyLinear(
      (linear): Linear(in_features=10, out_features=5, bias=True)
    )



```python
x = torch.FloatTensor(16, 10)
print(x)
```

    tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
             0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
             0.0000e+00, 0.0000e+00, 2.5857e-35, 4.5901e-41],
            [2.5857e-35, 4.5901e-41, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
             0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
             0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
             0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
             0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
             0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
             0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
             0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
             0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
             0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
             0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
             0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
             0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
             0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
             0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]])



```python
objective = 100

linear = MyLinear(10, 5)
y = linear(x)
print(y)
print(y.shape)
```

    tensor([[-0.2852, -0.1867, -0.2791, -0.0219, -0.1018],
            [-0.2852, -0.1867, -0.2791, -0.0219, -0.1018],
            [-0.2852, -0.1867, -0.2791, -0.0219, -0.1018],
            [-0.2852, -0.1867, -0.2791, -0.0219, -0.1018],
            [-0.2852, -0.1867, -0.2791, -0.0219, -0.1018],
            [-0.2852, -0.1867, -0.2791, -0.0219, -0.1018],
            [-0.2852, -0.1867, -0.2791, -0.0219, -0.1018],
            [-0.2852, -0.1867, -0.2791, -0.0219, -0.1018],
            [-0.2852, -0.1867, -0.2791, -0.0219, -0.1018],
            [-0.2852, -0.1867, -0.2791, -0.0219, -0.1018],
            [-0.2852, -0.1867, -0.2791, -0.0219, -0.1018],
            [-0.2852, -0.1867, -0.2791, -0.0219, -0.1018],
            [-0.2852, -0.1867, -0.2791, -0.0219, -0.1018],
            [-0.2852, -0.1867, -0.2791, -0.0219, -0.1018],
            [-0.2852, -0.1867, -0.2791, -0.0219, -0.1018],
            [-0.2852, -0.1867, -0.2791, -0.0219, -0.1018]],
           grad_fn=<AddmmBackward0>)
    torch.Size([16, 5])



```python
loss = abs(objective - y.sum()) # Absolute Error
print(loss)

loss.backward() # gt에서 y를 뺀 것이므로 역시 추적 대상, 여기서의 미분값을 계산
```

    tensor(113.9951, grad_fn=<AbsBackward0>)



```python
# Training...
linear.eval()
# Do some inference process.
linear.train()
# Restart training, again.
```




    MyLinear(
      (linear): Linear(in_features=10, out_features=5, bias=True)
    )




```python
import random

import torch
import torch.nn as nn

class MyModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()

        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        y = self.linear(x)

        return y
```


```python
def ground_truth(x):
    return 3 * x[:, 0] + x[:, 1] - 2 * x[:, 2]
```


```python
def train(model, x, y, optim):
    # initialize gradients in all parameters in module.
    optim.zero_grad()# backward()는 계속 값을 누적하므로 매번 gradient를 0으로 만들어야 한다.

    # feed-forward
    y_hat = model(x)
    # get error between answer and inferenced.
    loss = ((y - y_hat)**2).sum() / x.size(0) # 직접 계산할 수도 있고 pytorch에서 제공하는 loss function을 사용할 수도 있다.

    # back-propagation
    loss.backward()

    # one-step of gradient descent
    optim.step()

    return loss.data
```


```python
batch_size = 1
n_epochs = 1000
n_iter = 10000

model = MyModel(3, 1)
optim = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.1)

print(model)
```

    MyModel(
      (linear): Linear(in_features=3, out_features=1, bias=True)
    )



```python
for epoch in range(n_epochs):
    avg_loss = 0

    model.train()
    for i in range(n_iter):
        x = torch.rand(batch_size, 3)
        y = ground_truth(x.data)

        loss = train(model, x, y, optim)

        avg_loss += loss
    avg_loss = avg_loss / n_iter

    # simple test sample to check the network.
    x_valid = torch.FloatTensor([[.3, .2, .1]])
    y_valid = ground_truth(x_valid.data)

    model.eval()
    y_hat = model(x_valid)

    print(avg_loss, y_valid.data[0], y_hat.data[0, 0])

    if avg_loss < .1: # finish the training if the loss is smaller than .1.
        break
```

    tensor(1.1094) tensor(0.9000) tensor(0.8250)
    tensor(0.7934) tensor(0.9000) tensor(0.8800)
    tensor(0.5385) tensor(0.9000) tensor(0.9051)
    tensor(0.3762) tensor(0.9000) tensor(0.9288)
    tensor(0.2662) tensor(0.9000) tensor(0.9454)
    tensor(0.1820) tensor(0.9000) tensor(0.9584)
    tensor(0.1286) tensor(0.9000) tensor(0.9637)
    tensor(0.0900) tensor(0.9000) tensor(0.9687)



```python
# Note that tensor is declared in torch.cuda.
#x = torch.cuda.FloatTensor(16, 10)
#linear = MyLinear(10, 5)
# .cuda() let module move to GPU memory.
#linear.cuda()
#y = linear(x)
```

## torch application basic


```python
# import 패키지

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
```


```python
class MyModel(nn.Module):
    def __init__(self, X_dim, y_dim):
        super(MyModel, self).__init__()
        layer1 = nn.Linear(X_dim, 128)
        activation1 = nn.ReLU()
        layer2 = nn.Linear(128, y_dim)
        self.module = nn.Sequential(
            layer1,
            activation1,
            layer2
        )
        
    def forward(self, x):
        out = self.module(x)
        result = F.softmax(out, dim=1)
        return result        
```


```python
## pytorch 어플리케이션의 기본 구조

#for input, target in dataset:
#    optimizer.zero_grad()
#    output = model(input)
#    loss = loss_fn(output, target)
#    loss.backward()
#    optimizer.step()
```
