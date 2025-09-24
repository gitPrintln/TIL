## [QMNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#qmnist)


```python
%pip install torchvision
```

    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: torchvision in /home/ubuntu/.local/lib/python3.10/site-packages (0.22.1)
    Requirement already satisfied: torch==2.7.1 in /home/ubuntu/.local/lib/python3.10/site-packages (from torchvision) (2.7.1)
    Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /home/ubuntu/.local/lib/python3.10/site-packages (from torchvision) (11.3.0)
    Requirement already satisfied: numpy in /home/ubuntu/.local/lib/python3.10/site-packages (from torchvision) (2.2.6)
    Requirement already satisfied: nvidia-cusparse-cu12==12.5.4.2 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch==2.7.1->torchvision) (12.5.4.2)
    Requirement already satisfied: nvidia-nvtx-cu12==12.6.77 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch==2.7.1->torchvision) (12.6.77)
    Requirement already satisfied: nvidia-cuda-cupti-cu12==12.6.80 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch==2.7.1->torchvision) (12.6.80)
    Requirement already satisfied: nvidia-cudnn-cu12==9.5.1.17 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch==2.7.1->torchvision) (9.5.1.17)
    Requirement already satisfied: nvidia-cusolver-cu12==11.7.1.2 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch==2.7.1->torchvision) (11.7.1.2)
    Requirement already satisfied: nvidia-nccl-cu12==2.26.2 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch==2.7.1->torchvision) (2.26.2)
    Requirement already satisfied: networkx in /home/ubuntu/.local/lib/python3.10/site-packages (from torch==2.7.1->torchvision) (3.4.2)
    Requirement already satisfied: jinja2 in /usr/lib/python3/dist-packages (from torch==2.7.1->torchvision) (3.0.3)
    Requirement already satisfied: triton==3.3.1 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch==2.7.1->torchvision) (3.3.1)
    Requirement already satisfied: nvidia-cuda-runtime-cu12==12.6.77 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch==2.7.1->torchvision) (12.6.77)
    Requirement already satisfied: nvidia-curand-cu12==10.3.7.77 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch==2.7.1->torchvision) (10.3.7.77)
    Requirement already satisfied: filelock in /home/ubuntu/.local/lib/python3.10/site-packages (from torch==2.7.1->torchvision) (3.18.0)
    Requirement already satisfied: fsspec in /home/ubuntu/.local/lib/python3.10/site-packages (from torch==2.7.1->torchvision) (2025.7.0)
    Requirement already satisfied: typing-extensions>=4.10.0 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch==2.7.1->torchvision) (4.14.1)
    Requirement already satisfied: nvidia-cufft-cu12==11.3.0.4 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch==2.7.1->torchvision) (11.3.0.4)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.6.77 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch==2.7.1->torchvision) (12.6.77)
    Requirement already satisfied: sympy>=1.13.3 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch==2.7.1->torchvision) (1.14.0)
    Requirement already satisfied: nvidia-cusparselt-cu12==0.6.3 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch==2.7.1->torchvision) (0.6.3)
    Requirement already satisfied: nvidia-nvjitlink-cu12==12.6.85 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch==2.7.1->torchvision) (12.6.85)
    Requirement already satisfied: nvidia-cublas-cu12==12.6.4.1 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch==2.7.1->torchvision) (12.6.4.1)
    Requirement already satisfied: nvidia-cufile-cu12==1.11.1.6 in /home/ubuntu/.local/lib/python3.10/site-packages (from torch==2.7.1->torchvision) (1.11.1.6)
    Requirement already satisfied: setuptools>=40.8.0 in /usr/lib/python3/dist-packages (from triton==3.3.1->torch==2.7.1->torchvision) (59.6.0)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /home/ubuntu/.local/lib/python3.10/site-packages (from sympy>=1.13.3->torch==2.7.1->torchvision) (1.3.0)
    Note: you may need to restart the kernel to use updated packages.



```python
import os
import random
import numpy as np
import torch
import torchvision
```


```python
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
```


```python
dataset = torchvision.datasets.QMNIST('./data', download=True)
```


```python
print(type(dataset))

# tuple: (image, target)
```

    <class 'torchvision.datasets.mnist.QMNIST'>



```python
print(len(dataset))
```

    60000



```python
n = 200
data, target = dataset[n]
```


```python
print(type(data))
```

    <class 'PIL.Image.Image'>



```python
print(data.mode, data.width, data.height)
```

    L 28 28



```python
print(type(target))
```

    <class 'int'>



```python
import matplotlib.pyplot as plt
%matplotlib inline

plt.title(target)
plt.imshow(data, cmap='gray')
data.show()
```

    /home/ubuntu/.local/lib/python3.10/site-packages/numpy/_core/getlimits.py:551: UserWarning: Signature b'\x00\xd0\xcc\xcc\xcc\xcc\xcc\xcc\xfb\xbf\x00\x00\x00\x00\x00\x00' for <class 'numpy.longdouble'> does not match any known type: falling back to type probe function.
    This warnings indicates broken support for the dtype!
      machar = _get_machar(dtype)



    
![png](QMNIST3_files/QMNIST3_11_1.png)
    



    
![png](QMNIST3_files/QMNIST3_11_2.png)
    



```python
import torchvision.transforms as transforms
# batch must contain tensors, numpy arrays, numbers, dicts or lists
ToTensor = transforms.Compose([
    transforms.ToTensor()
])
dataset = torchvision.datasets.QMNIST('./data', transform = ToTensor)
```


```python
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=1,
                                          shuffle=True)
```


```python
ToPILImage = transforms.Compose([
    transforms.ToPILImage()
])

for data, target in data_loader:
    print(data.shape)
    data = data.squeeze() # 불필요한 차원 하나를 줄인다(batch_size=1)
    data = data.squeeze() # 불필요한 차원 하나를 줄인다(color)
    print(data.shape)
    pic = ToPILImage(data)
    plt.title(int(target))
    plt.imshow(pic, cmap='gray')
    plt.show()
    break
```

    torch.Size([1, 1, 28, 28])
    torch.Size([28, 28])



    
![png](QMNIST3_files/QMNIST3_14_1.png)
    



```python
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=10,
                                          shuffle=True)
```


```python
ToPILImage = transforms.Compose([
    transforms.ToPILImage()
])

for data, target in data_loader:
    index = 5
    print(data.shape)
    img = data[index]
    print(img.shape)
    img = img[0] 
    print(img.shape)
    pic = ToPILImage(img)
    plt.title(int(target[index]))
    plt.imshow(img, cmap='gray')
    plt.show()
    break
```

    torch.Size([10, 1, 28, 28])
    torch.Size([1, 28, 28])
    torch.Size([28, 28])



    
![png](QMNIST3_files/QMNIST3_16_1.png)
    



```python
# 1000개 batch
n = 1000
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=n,
                                          shuffle=True)
i = 0
l = []
for data, target in data_loader:
    i += len(data)
    l.append(len(data))
    
print(l)
print('Total number of data: {}'.format(i))

# 2000개 batch
n = 2000
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=n,
                                          shuffle=True)
i = 0
l = []
for data, target in data_loader:
    i += len(data)
    l.append(len(data))
    
print(l)
print('Total number of data: {}'.format(i))

# 999개 batch
n = 999
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=n,
                                          shuffle=True)
i = 0
l = []
for data, target in data_loader:
    i += len(data)
    l.append(len(data))

print(l)
print('Total number of data: {}'.format(i))
```

    [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    Total number of data: 60000
    [2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000]
    Total number of data: 60000
    [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 60]
    Total number of data: 60000



```python
%matplotlib inline

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as utils
import torchvision.transforms as transforms

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')
```


```python
print(is_cuda)
print(device)
```

    False
    cpu



```python
import torchvision.transforms as transforms
compose = transforms.Compose([
    transforms.ToTensor()
])

train_data = torchvision.datasets.QMNIST(root='./data/', train=True, transform=compose, download=True)
test_data  = torchvision.datasets.QMNIST(root='./data/', train=False, transform=compose, download=True)
```

    100.0%
    100.0%



```python
print('train sets: {}'.format(len(train_data)))
print('test sets: {}'.format(len(test_data)))
```

    train sets: 60000
    test sets: 60000



```python
BATCH_SIZE = 10

train_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)
```


```python
class MNISTModel(nn.Module):
    def __init__(self, output_size):
        super(MNISTModel, self).__init__()
        self.cv = nn.Sequential( # (n, 1, 28, 28)
                nn.Conv2d(1, 8, kernel_size=3), # 1 = input channel size(gray), 8 = output(filter) volume size (n, 8, 26, 26)
                                              # padding = 0, stride = (1, 1)
                                # (n, 8, 26, 26)
                nn.ReLU(),
                nn.MaxPool2d(2), # (n, 8, 13, 13)
                nn.Conv2d(8, 10, kernel_size=4), # (n, 10, 10, 10)
                nn.ReLU(),
                nn.MaxPool2d(2) # (n, 10, 5, 5)
        )

        self.fc = nn.Sequential(nn.Linear(10*5*5, output_size), # (n, 10*5*5)
              )        
        
    def forward(self, x):
        x = self.cv(x)
        x = x.view(-1, 10*5*5)
        output = self.fc(x)
        value = torch.max(output, 1)[1]
        return output, value
```


```python
model = MNISTModel(10).to(device)
```


```python
print(model)
```

    MNISTModel(
      (cv): Sequential(
        (0): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1))
        (1): ReLU()
        (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (3): Conv2d(8, 10, kernel_size=(4, 4), stride=(1, 1))
        (4): ReLU()
        (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (fc): Sequential(
        (0): Linear(in_features=250, out_features=10, bias=True)
      )
    )



```python
print(sum(p.numel() for p in model.parameters()))
```

    3880



```python
n = 0
for data, target in train_loader:
    print(data.shape) # (batch, channel, width, height)
    break

data = data.to(device)

with torch.no_grad():
    y, _ = model(data)
    print(y)
    print(np.argmax(y.cpu(), axis=1), target)
```

    torch.Size([10, 1, 28, 28])
    tensor([[-0.0197, -0.0144,  0.0113, -0.0123,  0.0556, -0.0263, -0.0356, -0.0422,
             -0.0966, -0.0090],
            [-0.0040, -0.0192,  0.0179, -0.0012,  0.0631, -0.0646, -0.0354, -0.0231,
             -0.0934, -0.0314],
            [ 0.0020,  0.0201,  0.0389, -0.0224,  0.0724, -0.0463, -0.0540, -0.0348,
             -0.1426, -0.0479],
            [-0.0347, -0.0087, -0.0003, -0.0151,  0.0774, -0.0346, -0.0395, -0.0321,
             -0.1051,  0.0221],
            [-0.0062, -0.0281,  0.0037, -0.0313,  0.0827, -0.0343, -0.0329, -0.0238,
             -0.1107, -0.0064],
            [-0.0049, -0.0027,  0.0089, -0.0200,  0.0662, -0.0454, -0.0641, -0.0212,
             -0.1251,  0.0051],
            [-0.0182, -0.0124,  0.0196, -0.0157,  0.0524, -0.0220, -0.0523,  0.0002,
             -0.0950, -0.0308],
            [-0.0237, -0.0031,  0.0241, -0.0176,  0.0714, -0.0294, -0.0317, -0.0200,
             -0.1262, -0.0191],
            [ 0.0104, -0.0109,  0.0015, -0.0291,  0.0663, -0.0471, -0.0617, -0.0170,
             -0.1333, -0.0154],
            [ 0.0043,  0.0066,  0.0156, -0.0398,  0.0914, -0.0628, -0.0550, -0.0340,
             -0.1331,  0.0080]])
    tensor([4, 4, 4, 4, 4, 4, 4, 4, 4, 4]) tensor([9, 1, 5, 4, 4, 6, 0, 2, 6, 6])



```python
BATCH_SIZE = 1000

train_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)
```


```python
model = MNISTModel(10).to(device) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

n_epochs = 5 # 5회 학습

list_training_loss = []
list_test_loss = []

for epoch in range(n_epochs):
    n_train = 0
    train_loss = 0
    model.train()
    for train_data, train_target in train_loader:
        train_data = train_data.to(device)
        train_target = train_target.to(device)
        y, _ = model(train_data)
        loss = criterion(y, train_target).sum()
        train_loss += loss.data.cpu().numpy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n_train += 1
        del train_data
        del train_target

    model.eval()
    n_test = 0
    test_loss = 0
    correct = 0
    total = 0
    for test_data, test_target in test_loader:
        test_data = test_data.to(device)
        test_target = test_target.to(device)
        y_pred, idx_pred = model(test_data)
        del test_data
        loss = criterion(y_pred, test_target).sum()
        test_loss += loss.data.cpu().numpy()
        n_test += 1
        total += len(test_target)
        correct += (idx_pred == test_target).sum()

    accuracy = correct * 100 / total
    print('epoch {}th training loss: {} test loss: {}, accuracy: {}'.format(
            epoch, train_loss / n_train, test_loss / n_test,
            accuracy
        ))
    list_training_loss.append(train_loss / n_train)
    list_test_loss.append(test_loss / n_test)
```

    epoch 0th training loss: 1.6951669454574585 test loss: 0.7355597615242004, accuracy: 80.52833557128906
    epoch 1th training loss: 0.49404260516166687 test loss: 0.3665483593940735, accuracy: 89.41500091552734
    epoch 2th training loss: 0.32944542169570923 test loss: 0.28246697783470154, accuracy: 91.8550033569336
    epoch 3th training loss: 0.26499152183532715 test loss: 0.2346922606229782, accuracy: 93.27666473388672
    epoch 4th training loss: 0.22518227994441986 test loss: 0.2031428962945938, accuracy: 94.04000091552734



```python
import matplotlib.pyplot as plt

plt.plot(list_training_loss, label='training')
plt.plot(list_test_loss, label='test')
plt.legend()
plt.show()
```


    
![png](QMNIST3_files/QMNIST3_30_0.png)
    



```python
from sklearn.metrics import accuracy_score
import numpy as np

model.eval()
y_test = None
y_pred = None
for test_data, test_target in test_loader:
    test_data = test_data.to(device)
    test_target = test_target.to(device)
    _, y = model(test_data)
    del test_data
    if None == y_test:
        y_test = test_target
    else:
        torch.cat([y_test, test_target], dim=0)
    if None == y_pred:
        y_pred = y
    else:
        torch.cat([y_pred, y], dim=0)

score = accuracy_score(y_test.cpu(), y_pred.cpu())
print(score)
```

    0.934

