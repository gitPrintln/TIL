```python
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
```

    /home/ubuntu/.local/lib/python3.10/site-packages/numpy/_core/getlimits.py:551: UserWarning: Signature b'\x00\xd0\xcc\xcc\xcc\xcc\xcc\xcc\xfb\xbf\x00\x00\x00\x00\x00\x00' for <class 'numpy.longdouble'> does not match any known type: falling back to type probe function.
    This warnings indicates broken support for the dtype!
      machar = _get_machar(dtype)



```python
df = sns.load_dataset('diamonds')
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
      <th>carat</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.23</td>
      <td>Ideal</td>
      <td>E</td>
      <td>SI2</td>
      <td>61.5</td>
      <td>55.0</td>
      <td>326</td>
      <td>3.95</td>
      <td>3.98</td>
      <td>2.43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.21</td>
      <td>Premium</td>
      <td>E</td>
      <td>SI1</td>
      <td>59.8</td>
      <td>61.0</td>
      <td>326</td>
      <td>3.89</td>
      <td>3.84</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.23</td>
      <td>Good</td>
      <td>E</td>
      <td>VS1</td>
      <td>56.9</td>
      <td>65.0</td>
      <td>327</td>
      <td>4.05</td>
      <td>4.07</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.29</td>
      <td>Premium</td>
      <td>I</td>
      <td>VS2</td>
      <td>62.4</td>
      <td>58.0</td>
      <td>334</td>
      <td>4.20</td>
      <td>4.23</td>
      <td>2.63</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.31</td>
      <td>Good</td>
      <td>J</td>
      <td>SI2</td>
      <td>63.3</td>
      <td>58.0</td>
      <td>335</td>
      <td>4.34</td>
      <td>4.35</td>
      <td>2.75</td>
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
      <th>carat</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>53935</th>
      <td>0.72</td>
      <td>Ideal</td>
      <td>D</td>
      <td>SI1</td>
      <td>60.8</td>
      <td>57.0</td>
      <td>2757</td>
      <td>5.75</td>
      <td>5.76</td>
      <td>3.50</td>
    </tr>
    <tr>
      <th>53936</th>
      <td>0.72</td>
      <td>Good</td>
      <td>D</td>
      <td>SI1</td>
      <td>63.1</td>
      <td>55.0</td>
      <td>2757</td>
      <td>5.69</td>
      <td>5.75</td>
      <td>3.61</td>
    </tr>
    <tr>
      <th>53937</th>
      <td>0.70</td>
      <td>Very Good</td>
      <td>D</td>
      <td>SI1</td>
      <td>62.8</td>
      <td>60.0</td>
      <td>2757</td>
      <td>5.66</td>
      <td>5.68</td>
      <td>3.56</td>
    </tr>
    <tr>
      <th>53938</th>
      <td>0.86</td>
      <td>Premium</td>
      <td>H</td>
      <td>SI2</td>
      <td>61.0</td>
      <td>58.0</td>
      <td>2757</td>
      <td>6.15</td>
      <td>6.12</td>
      <td>3.74</td>
    </tr>
    <tr>
      <th>53939</th>
      <td>0.75</td>
      <td>Ideal</td>
      <td>D</td>
      <td>SI2</td>
      <td>62.2</td>
      <td>55.0</td>
      <td>2757</td>
      <td>5.83</td>
      <td>5.87</td>
      <td>3.64</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 53940 entries, 0 to 53939
    Data columns (total 10 columns):
     #   Column   Non-Null Count  Dtype   
    ---  ------   --------------  -----   
     0   carat    53940 non-null  float64 
     1   cut      53940 non-null  category
     2   color    53940 non-null  category
     3   clarity  53940 non-null  category
     4   depth    53940 non-null  float64 
     5   table    53940 non-null  float64 
     6   price    53940 non-null  int64   
     7   x        53940 non-null  float64 
     8   y        53940 non-null  float64 
     9   z        53940 non-null  float64 
    dtypes: category(3), float64(6), int64(1)
    memory usage: 3.0 MB



```python
df.isnull().sum()
```




    carat      0
    cut        0
    color      0
    clarity    0
    depth      0
    table      0
    price      0
    x          0
    y          0
    z          0
    dtype: int64




```python
sns.pairplot(df)
```




    <seaborn.axisgrid.PairGrid at 0x7f8823dd8310>




    
![png](diamonds_files/diamonds_6_1.png)
    



```python
sns.barplot(x='cut', y='carat', data=df)
```




    <Axes: xlabel='cut', ylabel='carat'>




    
![png](diamonds_files/diamonds_7_1.png)
    



```python
sns.barplot(x='cut', y='price', data=df)
```




    <Axes: xlabel='cut', ylabel='price'>




    
![png](diamonds_files/diamonds_8_1.png)
    



```python
from pandas.api.types import is_categorical_dtype, is_object_dtype, is_numeric_dtype

def agg_func(col):
    if is_categorical_dtype(col) or is_object_dtype(col):
        return 'count'
    elif is_numeric_dtype(col):
        return 'mean'
    else:
        return 'first'
```


```python
columns_to_agg = [col for col in df.columns if col != 'cut']

groupedvalues = df.groupby('cut').agg({col: agg_func(df[col]) for col in columns_to_agg}).reset_index()
groupedvalues.head()
```

    /tmp/ipykernel_864/3343032604.py:3: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      groupedvalues = df.groupby('cut').agg({col: agg_func(df[col]) for col in columns_to_agg}).reset_index()
    /tmp/ipykernel_864/1889268883.py:4: DeprecationWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, pd.CategoricalDtype) instead
      if is_categorical_dtype(col) or is_object_dtype(col):





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
      <th>cut</th>
      <th>carat</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ideal</td>
      <td>0.702837</td>
      <td>21551</td>
      <td>21551</td>
      <td>61.709401</td>
      <td>55.951668</td>
      <td>3457.541970</td>
      <td>5.507451</td>
      <td>5.520080</td>
      <td>3.401448</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Premium</td>
      <td>0.891955</td>
      <td>13791</td>
      <td>13791</td>
      <td>61.264673</td>
      <td>58.746095</td>
      <td>4584.257704</td>
      <td>5.973887</td>
      <td>5.944879</td>
      <td>3.647124</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Very Good</td>
      <td>0.806381</td>
      <td>12082</td>
      <td>12082</td>
      <td>61.818275</td>
      <td>57.956150</td>
      <td>3981.759891</td>
      <td>5.740696</td>
      <td>5.770026</td>
      <td>3.559801</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Good</td>
      <td>0.849185</td>
      <td>4906</td>
      <td>4906</td>
      <td>62.365879</td>
      <td>58.694639</td>
      <td>3928.864452</td>
      <td>5.838785</td>
      <td>5.850744</td>
      <td>3.639507</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fair</td>
      <td>1.046137</td>
      <td>1610</td>
      <td>1610</td>
      <td>64.041677</td>
      <td>59.053789</td>
      <td>4358.757764</td>
      <td>6.246894</td>
      <td>6.182652</td>
      <td>3.982770</td>
    </tr>
  </tbody>
</table>
</div>




```python
g =sns.barplot(x='cut',y='price',data=groupedvalues)

for index, row in groupedvalues.iterrows():
    g.text(row.name, row.price + 80, round(row.price, 2), color='black', ha="center")
```


    
![png](diamonds_files/diamonds_11_0.png)
    



```python
plt.figure(figsize=(12, 5))
sns.barplot(x='cut', y='price', hue = 'color', data=df)
```




    <Axes: xlabel='cut', ylabel='price'>




    
![png](diamonds_files/diamonds_12_1.png)
    



```python
plt.figure(figsize=(12, 5))
sns.barplot(x='cut', y='price', hue = 'color', order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], data=df)
```




    <Axes: xlabel='cut', ylabel='price'>




    
![png](diamonds_files/diamonds_13_1.png)
    



```python
plt.figure(figsize=(12, 5))
sns.barplot(x='cut', y='price', hue = 'color', order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], estimator=sum, data=df)
```




    <Axes: xlabel='cut', ylabel='price'>




    
![png](diamonds_files/diamonds_14_1.png)
    



```python
plt.figure(figsize=(12, 5))
sns.barplot(x='cut', y='price', hue = 'color', order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], palette="Blues_d", estimator=sum, data=df)
```




    <Axes: xlabel='cut', ylabel='price'>




    
![png](diamonds_files/diamonds_15_1.png)
    



```python
sns.catplot(x='cut', y='price', hue = 'color', col="clarity", data=df, kind="bar")
```




    <seaborn.axisgrid.FacetGrid at 0x7f8804bb7f40>




    
![png](diamonds_files/diamonds_16_1.png)
    



```python
group = df.groupby(['cut', 'clarity']) 
g = group.size().unstack() 
plt.figure(figsize=(12, 5))
sns.heatmap(g, annot = True)
```

    /tmp/ipykernel_864/2168984754.py:1: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      group = df.groupby(['cut', 'clarity'])





    <Axes: xlabel='clarity', ylabel='cut'>




    
![png](diamonds_files/diamonds_17_2.png)
    



```python
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

dfi = df.copy()

dfi['cut'] = label_encoder.fit_transform(dfi['cut'])
print(label_encoder.classes_)
dfi['color'] = label_encoder.fit_transform(dfi['color'])
print(label_encoder.classes_)
dfi['clarity'] = label_encoder.fit_transform(dfi['clarity'])
print(label_encoder.classes_)
```

    ['Fair' 'Good' 'Ideal' 'Premium' 'Very Good']
    ['D' 'E' 'F' 'G' 'H' 'I' 'J']
    ['I1' 'IF' 'SI1' 'SI2' 'VS1' 'VS2' 'VVS1' 'VVS2']



```python
dfi.head()
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
      <th>carat</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.23</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>61.5</td>
      <td>55.0</td>
      <td>326</td>
      <td>3.95</td>
      <td>3.98</td>
      <td>2.43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.21</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>59.8</td>
      <td>61.0</td>
      <td>326</td>
      <td>3.89</td>
      <td>3.84</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.23</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>56.9</td>
      <td>65.0</td>
      <td>327</td>
      <td>4.05</td>
      <td>4.07</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.29</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>62.4</td>
      <td>58.0</td>
      <td>334</td>
      <td>4.20</td>
      <td>4.23</td>
      <td>2.63</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.31</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>63.3</td>
      <td>58.0</td>
      <td>335</td>
      <td>4.34</td>
      <td>4.35</td>
      <td>2.75</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pairplot(dfi)
```




    <seaborn.axisgrid.PairGrid at 0x7f87fe5d2e30>




    
![png](diamonds_files/diamonds_20_1.png)
    

