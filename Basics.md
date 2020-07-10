# How to start a machine learning project from scratch?  
**Contents**  
gather the data  
prepare the data  
select model  
train the model  
evaluate the model  
tune parameters  
get prediction(implementation)  

## Gather Data  
根据需要采集数据，过程中可能发生数据缺失或者受到污染的问题。  
## Prepare Data  
### Exploration Data Analysis  
拿到数据第一步需要对数据全貌，基本特性，各项特征的内容有一定的了解。  
1. 描述  
```python
import pandas as pd
data = pd.read_csv(fpath, index_col= , parse_dates= )
# 基本信息
data.head()
data.dtypes() # string类型的数据也会被归入'object'类型中
data.describe() # 基础统计信息
# 缺失值
data.isnull().sum() / len(data)
```
2. 选择部分直观上觉得有用的特征  
用少量特征和简单预测模型建立一个baseline  
3. baseline model上数据的进一步探索  
此部分用pandas库可以极大简化操作，提高数据分析效率，一些常用操作见笔记[]  
### Feature Engineering  
* Missing Values  
出现原因:1.调查对象不愿提供 2.遗漏 3.数据逻辑上的先后关系，有的数据只能在特定情况下才有  
```python
# ---- drop ----
data.isnull.sum() / len(data)  # 缺失比例
data.dropna(axis= )
data.drop([])

# ---- imputation ----
from sklearn.impute import SimpleImputer
# SimpleImputer is based on mean value imputation
imputer = SimpleImputer()
train_impute = imputer.fit_transform(train_x)
test_impute = imputer.fit_transform(test_x)
pd.DataFrame(train_impute, index= , columns= )
...

# ---- imputation extension ----
# 数据项缺失本身可能就包含了一些信息，有时候是对预测有用的，所以在补全之外加上某特征是否有缺失的表示
data[col+'_bool'] = data[col].isnull()
imputation...
```
* Categorival Variables  
```python
# ---- label encoding ----
# 序列型类别ordinal variables，用有序数值打标签可以反应出它们的先后顺序，在诸如决策树模型中很有用
from sklearn.preprocessing import LabelEncoder
...

# ---- one-hot encoding ----
# 类别之间没有先后顺序，nominal variables
```
