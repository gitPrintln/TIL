## fetch_convtype


```python
from sklearn.datasets import fetch_covtype
import pandas as pd
import seaborn as sns
%matplotlib inline
```


```python
data = fetch_covtype()
```


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
```


```python
class CovTypeModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(CovTypeModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        
    def forward(self, x):
        output = self.model(x)
        with torch.no_grad():
            value = torch.argmax(output, dim=1)
        return output, value        
```


```python
model = CovTypeModel(54, 7)
```


```python
import numpy as np
from sklearn.model_selection import train_test_split

X = torch.Tensor(data.data)
y = torch.LongTensor(data.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # train과 test를 8:2로 분할
```


```python
model = CovTypeModel(54, 7)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

n_epochs = 1000 # 1000회 학습

avg_loss = 0 
for epoch in range(n_epochs):
    y, _ = model(X_train)
    target = y_train - 1
    loss = criterion(y, target).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('epoch {}th loss: {}'.format(epoch, loss.data))
```

    epoch 0th loss: 188.63113403320312
    epoch 100th loss: 1.928119421005249
    epoch 200th loss: 1.8216912746429443
    epoch 300th loss: 1.2706258296966553
    epoch 400th loss: 0.9945950508117676
    epoch 500th loss: 1.2824770212173462
    epoch 600th loss: 1.3514597415924072
    epoch 700th loss: 2.127744197845459
    epoch 800th loss: 0.7784491181373596
    epoch 900th loss: 1.2335985898971558



```python
n = 99
with torch.no_grad():
    y, _ = model(torch.unsqueeze(X_train[n], dim=0))
    print(y)
    print(y.sum())
    print(np.argmax(y), y_train[n]-1)
```


```python
model = CovTypeModel(54, 7)
optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
criterion = nn.CrossEntropyLoss()

n_epochs = 1000 # 1000회 학습

list_training_loss = []
list_test_loss = []

for epoch in range(n_epochs):
    model.train()
    y, _ = model(X_train)
    target = y_train - 1
    loss = criterion(y, target).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        model.eval()
        y_pred, _ = model(X_test)
        test_loss = criterion(y_pred, y_test - 1).sum()
        print('epoch {}th training loss: {} test loss: {}'.format(epoch, loss.data, test_loss.data))
        list_training_loss.append(loss.data)
        list_test_loss.append(test_loss.data)
```


```python
import matplotlib.pyplot as plt

plt.plot(list_training_loss, label='training')
plt.plot(list_test_loss, label='test')
plt.legend()
plt.show()
```


```python
from sklearn.metrics import accuracy_score

with torch.no_grad():
    _ , y_pred = model(X_test)
    score = accuracy_score(y_test-1, y_pred)
    print(score)
```
