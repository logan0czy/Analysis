# Exploring Data Analysis

实际问题中，在对问题进行建模调参求得拟合的模型之前，最重要的步骤是进行EDA，它决定了之后的特征工程的方向以及模型的训练效果。通过EDA，能够加深我们对收集到的数据的细节的认识。我认为数据的EDA可以从两个方面来看待，第一个方面是各项数据本身特性的分析，第二个方面是各项数据之间的相关性分析。

## Unique-各自特性

这一块中应该对训练集和测试集都进行分析

```python
#常用的数据分析及可视化的工具包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#导入数据，此处假设是.csv文件
train_data = pd.read_csv('data_path/train_file.csv', sep='')
test_data = pd.read_csv('data_path/test_file.csv', sep='')
```

### 1.脏数据，残缺的数据

* 数据加载检查

数据加载以后有必要简单看一下导入的数据大致的内容，并且检查一下有没有导入出错

```python
#训练集和测试集的头尾数据大致内容
train_data.head().append(train_data.tail())
test_data.head().append(test_data.tail())

#数据集的shape信息，对应数据量以及特征数（可能会包括label特征)
train_data.shape
test_data.shape
```

* 数据类型概览

常用的模型都是对数值进行处理的，pd.DataFrame中dtype为int, float的数据在之后有进一步的分析，这一步重点应该关照其他类型的数据，比如object类型的，因为在这些类型的数据中可能存在NaN以外的残缺值pandas不能识别（pandas只能识别并处理NaN残缺数据）

```python
# view dtype of each feature
train_data.info()
test_data.info()
```
### 2.基本统计特征

### 3.分布

## Correlation-相互之间的关联
