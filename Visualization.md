# 数据可视化常用操作概览  
主要来源: Kaggle [micro-course](https://www.kaggle.com/learn/data-visualization)  
```python
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
pd.plotting.register_maplotlib_converters()
import seaborn as sns
```
## Line Chart  
```python
plt.figure(figsize=(, ))
plt.title("...")

# 1. plot all features
sns.lineplot(data=df)
# 2. plot selected features
sns.lineplot(data=df[col], label="...")
sns.lineplot(data=df[col2], label="...")

plt.xlabel("...")
```
[可视化效果](https://www.kaggle.com/alexisbcook/line-charts)  
## Bar Chart & Heatmaps  
**Bar Chart**用于观察某列特征的数值分布情况，横轴可以是index或者其他特征  
```python
sns.barplot(x=df.index/df[col_other], y=df[col])
```
**Heatmaps**热力图，通过着色图像显示数据分布，感觉在特征相关性比较中很有用
```python
# set annot to 'True' to show value in each color cell
sns.heatmap(data=df, annot=True)
```
[可视化效果](https://www.kaggle.com/alexisbcook/bar-charts-and-heatmaps)  
## Scatter Plot  
```python
sns.scatterplot(x=df[col_1], y=df[col_2])
# or
sns.scatterplot(x='col_1', y='col_2', data=df)
```
添加线性拟合两特征之间的线性相关性  
```python
sns.regplot(x=df[col_1], y=df[col_2])
```
以第三个特征(类别特征)为集合划分依据，观察两个特征在不同集合下的关系  
```python
sns.scatterplot(x=df[col_1], y=df[col_2], hue=df[col_categorical])
```
**每一类别分别线性回归**  
```python
sns.lmplot(x='...', y='...', hue='...', data=df)
```
**Categorical Scatter Plot**画出某一个特征在某个类别特征下的数值分布散点图  
```python
sns.swarmplot(x=df[col], y=df[col_categorical])
```
[可视化效果](https://www.kaggle.com/alexisbcook/scatter-plots)
