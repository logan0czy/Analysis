# Exploring Data Analysis

实际问题中，在对问题进行建模调参求得拟合的模型之前，最重要的步骤是进行EDA，它决定了之后的特征工程的方向以及模型的训练效果。通过EDA，能够加深我们对收集到的数据的细节的认识。我认为数据的EDA可以从两个方面来看待，第一个方面是各项数据本身特性的分析，第二个方面是各项数据之间的相关性分析。

## 目录
* [本身特性分析](#Unique-各自特性)
  * [数据残缺情况](#数据残缺情况)
  * [基本统计特征](#基本统计特征)
  * [分布情况](#分布)
* [关联性分析](#Correlation-相互之间的关联)
* [生成数据报告](#生成数据报告)

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

### <span id='数据残缺情况'>1. 脏数据，残缺的数据</span>

#### 1.1 数据加载检查

数据加载以后有必要简单看一下导入的数据大致的内容，并且检查一下有没有导入出错  
```python
#训练集和测试集的头尾数据大致内容
train_data.head().append(train_data.tail())
test_data.head().append(test_data.tail())

#数据集的shape信息，对应数据量以及特征数（可能会包括label特征)
train_data.shape
test_data.shape
```

#### 1.2 数据类型概览

常用的模型都是对数值进行处理的，pd.DataFrame中dtype为int, float的数据在之后有进一步的分析，这一步重点应该关照其他类型的数据，比如object类型的，因为在这些类型的数据中可能存在NaN以外的残缺值pandas不能识别（pandas只能识别并处理NaN残缺数据）  
```python
# view dtype of each feature
train_data.info()
test_data.info()
```  
此时便可查看对应列数据的取值分布信息，观察有没有缺失值被处理成了一些符号，比如'-'等等。如果有的话可以将这些缺失部分统一替换为NaN方便后续的进一步处理，比如填充，删除等等。这里以二手车价格估计中的`notRepairedDamage`项为例。  
```python
train_data['notRepairedDamage'].value_counts()  # 该项数据下各个取值的统计 --> 输出发现0，1以外还有'-'，即表示的是缺失值
train_data['notRepairedDamage'].replace('-', np.nan, inplace=True)

# 同样的对测试集也有对应操作
test_data['notRepairedDamage'].value_counts()
test_data['notRepairedDamage'].replace('-', np.nan, inplace=True)
```

#### 1.3 缺省情况总览

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

### <span id='基本统计特征'>2. 基本统计特征</span>

对各项数据的取值范围以及一些基本的统计信息（个数count，均值mean，方差std，最小值min，中位数25%50%75%，最大值max）有基本的了解。  
```python
train_data.describe()
test_data.describe()
```

### <span id='分布'>3. 分布</span>

观察数据的分布目前总结出两个作用：一是确保训练集和测试集的数据分布大体一致，避免训练好的模型因为两个数据集的分布不一致而表现出比较差的性能；二是能看出倾斜数据，有的严重倾斜，对预测没有什么帮助，还可能产生干扰，有的具有一定倾斜程度，长尾效应等，这时可以使用对数变化等改变数据的分布特性，所以需要及早发现和处理避免影响模型的判断。  
此外，特征大一点说，可以分为**数值型特征**和**类别型特征**；小一点说可以分为**定类特征**、**定序特征**、**定距特征**。  
- 定类特征：仅仅是有类别编号，编号之间的相对值没有什么实际意义（比如地区编号）
- 定序特征：也是类别特征，但是类别编号的大小决定了对应数据的先后，高低等等（比如收入等级）
- 定距特征：数值型的，这个含义就很明显

不同的特征类型对应着不同的分析操作。

#### 3.1 分布可视化

可视化能对每个数据的分布情况有一个大体和直观的了解。  
```python
# --------数值型--------
f = pd.melt(Train_data, value_vars=columns_spec)
g = sns.FacetGrid(f, col="variable",  col_wrap=4, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")
```  
```python
# --------类别型--------
# 箱形图
for c in categorical_features:
    Train_data[c] = Train_data[c].astype('category')
    if Train_data[c].isnull().any():
        Train_data[c] = Train_data[c].cat.add_categories(['MISSING'])
        Train_data[c] = Train_data[c].fillna('MISSING')

def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)

f = pd.melt(Train_data, id_vars=['price'], value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(boxplot, "value", "price")

# 小提琴图
catg_list = categorical_features
target = 'price'
for catg in catg_list :
    sns.violinplot(x=catg, y=target, data=Train_data)
    plt.show()
    
# 柱形图
def bar_plot(x, y, **kwargs):
    sns.barplot(x=x, y=y)
    x=plt.xticks(rotation=90)

f = pd.melt(Train_data, id_vars=['price'], value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(bar_plot, "value", "price")

# 各自频数
def count_plot(x,  **kwargs):
    sns.countplot(x=x)
    x=plt.xticks(rotation=90)

f = pd.melt(Train_data,  value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(count_plot, "value")
```

#### 3.2 偏度和峰度

主要对数值型数据。偏度衡量数据总体取值分布的对称性，常说左偏、右偏；偏度描述数据取值分布形态的陡缓程度，常说高顶和平顶。两者都是以正态分布为基准来描述的。详见 [这里](https://support.minitab.com/zh-cn/minitab/18/help-and-how-to/statistics/basic-statistics/supporting-topics/data-concepts/how-skewness-and-kurtosis-affect-your-distribution/)  
```python
# 格式化打印出各项数据峰偏度值
for col in features:
    print('{:15}'.format(col), 
          'Skewness: {:05.2f}'.format(Train_data[col].skew()) , 
          ' '*5,
          'Kurtosis: {:06.2f}'.format(Train_data[col].kurt())  
          )
```

#### 3.3 单特征查看

如果要具体的查看某一个特征的分布，除了使用sns.distplot等可视化工具外，还可以打印出一些统计信息以便于分析。  
```python
train_data['col_spec'].unique()    # 查看有多少个不同的取值
train_data['col_spec'].value_counts()    # 各个取值的频数
```

#### 3.4 预测值分布

单独说这一块是因为分析预测值的分布情况有利于提高模型的拟合性能，比如预测值是长尾分布的话可以考虑用log变换来处理。  
```python
y = train_data['price']
# 1)偏度和峰度
y.skew()
y.kurt()

# 2)总体分布情况，用不同的分布进行拟合
import scipy.stats as st
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)

# 3)取值频数
plt.hist(y, orientation = 'vertical',histtype = 'bar', color ='red')
```

## Correlation-相互之间的关联

主要分为预测值和各特征之间的相关性、各特征相互之间的相关性，进行相关性的分析便于之后在特征工程中对各个特征进行相应的取舍和处理。（此处主要是针对数值特征，类别特征有待进一步考量确认）(关于多变量之间相互关系的可视化更多信息可以参考 [这里](https://www.jianshu.com/p/6e18d21a4cad))  
* 预测值和输入特征之间  
```python
price_numeric = Train_data[numeric_features]
correlation = price_numeric.corr()
print(correlation['price'].sort_values(ascending = False),'\n')
```  
```python
# 预测值和单个特征之间的回归关系
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(nrows=5, ncols=2, figsize=(24, 20))
# ['v_12', 'v_8' , 'v_0', 'power', 'v_5',  'v_2', 'v_6', 'v_1', 'v_14']
v_12_scatter_plot = pd.concat([Y_train,Train_data['v_12']],axis = 1)
sns.regplot(x='v_12',y = 'price', data = v_12_scatter_plot,scatter= True, fit_reg=True, ax=ax1)

v_8_scatter_plot = pd.concat([Y_train,Train_data['v_8']],axis = 1)
sns.regplot(x='v_8',y = 'price',data = v_8_scatter_plot,scatter= True, fit_reg=True, ax=ax2)

v_0_scatter_plot = pd.concat([Y_train,Train_data['v_0']],axis = 1)
sns.regplot(x='v_0',y = 'price',data = v_0_scatter_plot,scatter= True, fit_reg=True, ax=ax3)

power_scatter_plot = pd.concat([Y_train,Train_data['power']],axis = 1)
sns.regplot(x='power',y = 'price',data = power_scatter_plot,scatter= True, fit_reg=True, ax=ax4)

v_5_scatter_plot = pd.concat([Y_train,Train_data['v_5']],axis = 1)
sns.regplot(x='v_5',y = 'price',data = v_5_scatter_plot,scatter= True, fit_reg=True, ax=ax5)

v_2_scatter_plot = pd.concat([Y_train,Train_data['v_2']],axis = 1)
sns.regplot(x='v_2',y = 'price',data = v_2_scatter_plot,scatter= True, fit_reg=True, ax=ax6)

v_6_scatter_plot = pd.concat([Y_train,Train_data['v_6']],axis = 1)
sns.regplot(x='v_6',y = 'price',data = v_6_scatter_plot,scatter= True, fit_reg=True, ax=ax7)

v_1_scatter_plot = pd.concat([Y_train,Train_data['v_1']],axis = 1)
sns.regplot(x='v_1',y = 'price',data = v_1_scatter_plot,scatter= True, fit_reg=True, ax=ax8)

v_14_scatter_plot = pd.concat([Y_train,Train_data['v_14']],axis = 1)
sns.regplot(x='v_14',y = 'price',data = v_14_scatter_plot,scatter= True, fit_reg=True, ax=ax9)

v_13_scatter_plot = pd.concat([Y_train,Train_data['v_13']],axis = 1)
sns.regplot(x='v_13',y = 'price',data = v_13_scatter_plot,scatter= True, fit_reg=True, ax=ax10)
```  
* 输入特征之间  
```python
# 相互关系热力图，这里包含了预测变量price
f , ax = plt.subplots(figsize = (7, 7))
plt.title('Correlation of Numeric Features with Price',y=1,size=16)
sns.heatmap(correlation,square = True,  vmax=0.8)
```  
```python
# 相互关系可视化
sns.set()
columns = ['price', 'v_12', 'v_8' , 'v_0', 'power', 'v_5',  'v_2', 'v_6', 'v_1', 'v_14']
sns.pairplot(Train_data[columns],size = 2 ,kind ='scatter',diag_kind='kde')
plt.show()
```

## 生成数据报告

```python
import pandas_profiling
pfr = pandas_profiling.ProfileReport(Train_data)
pfr.to_file("./example.html")
```
