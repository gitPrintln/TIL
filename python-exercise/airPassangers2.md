```python
%pip install statsmodels matplotlib pandas
```

    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: statsmodels in /home/ubuntu/.local/lib/python3.10/site-packages (0.14.5)
    Requirement already satisfied: matplotlib in /home/ubuntu/.local/lib/python3.10/site-packages (3.10.5)
    Requirement already satisfied: pandas in /home/ubuntu/.local/lib/python3.10/site-packages (2.3.1)
    Requirement already satisfied: patsy>=0.5.6 in /home/ubuntu/.local/lib/python3.10/site-packages (from statsmodels) (1.0.1)
    Requirement already satisfied: numpy<3,>=1.22.3 in /home/ubuntu/.local/lib/python3.10/site-packages (from statsmodels) (2.2.6)
    Requirement already satisfied: scipy!=1.9.2,>=1.8 in /home/ubuntu/.local/lib/python3.10/site-packages (from statsmodels) (1.15.3)
    Requirement already satisfied: packaging>=21.3 in /home/ubuntu/.local/lib/python3.10/site-packages (from statsmodels) (25.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/lib/python3/dist-packages (from matplotlib) (2.4.7)
    Requirement already satisfied: contourpy>=1.0.1 in /home/ubuntu/.local/lib/python3.10/site-packages (from matplotlib) (1.3.2)
    Requirement already satisfied: pillow>=8 in /home/ubuntu/.local/lib/python3.10/site-packages (from matplotlib) (11.3.0)
    Requirement already satisfied: fonttools>=4.22.0 in /home/ubuntu/.local/lib/python3.10/site-packages (from matplotlib) (4.59.0)
    Requirement already satisfied: cycler>=0.10 in /home/ubuntu/.local/lib/python3.10/site-packages (from matplotlib) (0.12.1)
    Requirement already satisfied: kiwisolver>=1.3.1 in /home/ubuntu/.local/lib/python3.10/site-packages (from matplotlib) (1.4.8)
    Requirement already satisfied: python-dateutil>=2.7 in /home/ubuntu/.local/lib/python3.10/site-packages (from matplotlib) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /usr/lib/python3/dist-packages (from pandas) (2022.1)
    Requirement already satisfied: tzdata>=2022.7 in /home/ubuntu/.local/lib/python3.10/site-packages (from pandas) (2025.2)
    Requirement already satisfied: six>=1.5 in /usr/lib/python3/dist-packages (from python-dateutil>=2.7->matplotlib) (1.16.0)
    Note: you may need to restart the kernel to use updated packages.



```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
```

    /home/ubuntu/.local/lib/python3.10/site-packages/numpy/_core/getlimits.py:551: UserWarning: Signature b'\x00\xd0\xcc\xcc\xcc\xcc\xcc\xcc\xfb\xbf\x00\x00\x00\x00\x00\x00' for <class 'numpy.longdouble'> does not match any known type: falling back to type probe function.
    This warnings indicates broken support for the dtype!
      machar = _get_machar(dtype)



```python
# https://www.statsmodels.org/stable/datasets/index.html#available-datasets
```


```python
df = sm.datasets.get_rdataset("AirPassengers").data
```


```python
print(type(df))
```

    <class 'pandas.core.frame.DataFrame'>



```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1949.000000</td>
      <td>112</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1949.083333</td>
      <td>118</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1949.166667</td>
      <td>132</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1949.250000</td>
      <td>129</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1949.333333</td>
      <td>121</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>139</th>
      <td>1960.583333</td>
      <td>606</td>
    </tr>
    <tr>
      <th>140</th>
      <td>1960.666667</td>
      <td>508</td>
    </tr>
    <tr>
      <th>141</th>
      <td>1960.750000</td>
      <td>461</td>
    </tr>
    <tr>
      <th>142</th>
      <td>1960.833333</td>
      <td>390</td>
    </tr>
    <tr>
      <th>143</th>
      <td>1960.916667</td>
      <td>432</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df.shape)
```

    (144, 2)



```python
df['date'] = pd.to_datetime('1949-01-01') + pd.to_timedelta((df['time'] - 1949) * 365.25, unit='D')
df.set_index('date', inplace=True)
df.drop(columns='time', inplace=True)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>value</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1949-01-01 00:00:00.000000000</th>
      <td>112</td>
    </tr>
    <tr>
      <th>1949-01-31 10:29:59.999897149</th>
      <td>118</td>
    </tr>
    <tr>
      <th>1949-03-02 21:00:00.000102850</th>
      <td>132</td>
    </tr>
    <tr>
      <th>1949-04-02 07:30:00.000000000</th>
      <td>129</td>
    </tr>
    <tr>
      <th>1949-05-02 17:59:59.999897149</th>
      <td>121</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>144.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>280.298611</td>
    </tr>
    <tr>
      <th>std</th>
      <td>119.966317</td>
    </tr>
    <tr>
      <th>min</th>
      <td>104.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>180.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>265.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>360.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>622.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 144 entries, 1949-01-01 00:00:00 to 1960-12-01 13:30:00.000102859
    Data columns (total 1 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   value   144 non-null    int64
    dtypes: int64(1)
    memory usage: 2.2 KB



```python
df.isnull().sum()
```




    value    0
    dtype: int64




```python
df.dropna(inplace=True)
```


```python
df.plot()
```




    <Axes: xlabel='date'>




    
![png](airPassangers2_files/airPassangers2_14_1.png)
    



```python
# AR(AutoRegressive): 과거의 실제 관측값들로 현재를 예측
# MA(Moving Average): 과거의 예측 오차들로 현재를 예측
# ARMA = AR + MA
# 통계적 모델은 '데이터의 계절성과 추세가 뚜렷하다'라는 전제하에 사용되는 경량 모델
```


```python
df['rolling_mean'] = df['value'].rolling(window=12).mean() # 이동 평균: 추세 식별, 노이즈 제거, 계절성 분석을 위해 사용
df['rolling_std'] = df['value'].rolling(window=12).std() # 이동 표준편차: 시간에 따른 변동성의 변화를 추적. 증가하면 평균 주위 데이터의 변동성이 커진다는 의미

plt.figure(figsize=(10, 4))
plt.plot(df['value'], label='Original')
plt.plot(df['rolling_mean'], label='Rolling Mean')
plt.plot(df['rolling_std'], label='Rolling Std')
plt.legend()
plt.title('Rolling Mean & Std Dev')
plt.show()
```


    
![png](airPassangers2_files/airPassangers2_16_0.png)
    



```python
df['log_passengers'] = np.log(df['value'])
```


```python
df['diff1'] = df['log_passengers'].diff().dropna()
df['seasonal_diff'] = df['log_passengers'].diff(12).dropna()
```


```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(df['log_passengers'].diff().dropna(), lags=40) # k=40
plt.title('ACF - First Difference') # Autocorrelation Function. 시점 t의 값이 시점 t-k값과 얼마나 선형적으로 관계되어 있는가? 0~1
                                    # ACF가 급격히 감소하는 지점이 MA 차수
plt.show()
```


    
![png](airPassangers2_files/airPassangers2_19_0.png)
    

