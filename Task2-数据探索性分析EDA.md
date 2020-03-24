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

### 1. 脏数据，残缺的数据

### 1.1 数据加载检查

数据加载以后有必要简单看一下导入的数据大致的内容，并且检查一下有没有导入出错

```python
#训练集和测试集的头尾数据大致内容
train_data.head().append(train_data.tail())
test_data.head().append(test_data.tail())

#数据集的shape信息，对应数据量以及特征数（可能会包括label特征)
train_data.shape
test_data.shape
```

### 1.2 数据类型概览

常用的模型都是对数值进行处理的，pd.DataFrame中dtype为int, float的数据在之后有进一步的分析，这一步重点应该关照其他类型的数据，比如object类型的，因为在这些类型的数据中可能存在NaN以外的残缺值pandas不能识别（pandas只能识别并处理NaN残缺数据）

```python
# view dtype of each feature
train_data.info()
test_data.info()
```

此时便可查看对应列数据的取值分布信息，观察有没有缺失值被处理成了一些符号，比如'-'等等。如果有的话可以将这些缺失部分统一替换为NaN方便后续的进一步处理，比如填充，删除等等。这里以二手车价格估计中的'notRepairedDamage'项为例。

```python
train_data['notRepairedDamage'].value_counts()  # 该项数据下各个取值的统计 --> 输出发现0，1以外还有'-'，即表示的是缺失值
train_data['notRepairedDamage'].replace('-', np.nan, inplace=True)

# 同样的对测试集也有对应操作
test_data['notRepairedDamage'].value_counts()
test_data['notRepairedDamage'].replace('-', np.nan, inplace=True)
```

### 1.3 缺省情况总览

对每项特征下的数据缺失情况进行统计，对于这些数据，不同的模型有不同的处理方法。比如树模型就可以直接将这些项空缺，树模型可以自己优化；但是对于另外一些模型就需要进行一定的处理，比如缺失量太多时考虑直接将该项特征进行删除，缺失量不是很大时可以对缺失的部分进行填充（如均值，众数，中位数，0，-1等等）

```python
train_data.isnull().sum()
test_data.isnull().sum()
```

此外，在这一块下还可以看看出现缺失的特征项之间会不会有一定的关联，即相互之间是在同一条数据里面出现缺失的还是各自的缺失分布比较独立没啥明显的相关关系。

```python
import missingno as msno
msno.matrix(train_data.sample(250))
msno.matrix(test_data.sample(250))
```

### 2.基本统计特征

对各项数据的取值范围以及一些基本的统计信息（个数count，均值mean，方差std，最小值min，中位数25%50%75%，最大值max）有基本的了解。

```python
train_data.describe()
test_data.describe()
```

### 3.分布

观察数据的分布目前总结出两个作用：一是确保训练集和测试集的数据分布大体一致，避免训练好的模型因为两个数据集的分布不一致而表现出比较差的性能；二是能看出倾斜数据，对预测没有什么帮助，还可能产生干扰，所以需要及早发现和处理避免影响模型的判断。


## Correlation-相互之间的关联
