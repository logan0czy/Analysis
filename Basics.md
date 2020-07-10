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
1.描述  
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
2.选择部分直观上觉得有用的特征  
用少量特征和简单预测模型建立一个baseline  

3.baseline model上数据的进一步探索  
此部分用pandas库可以极大简化操作，提高数据分析效率，一些常用操作见笔记[]  
### Feature Engineering  
> 总体遵循--预处理操作将训练集和验证/测试集分离，避免data leakage问题。最好的方式make your data pipeline, 所有的预处理都放在pipeline内部进行  

**Missing Values**  
出现原因:  1.调查对象不愿提供  2.遗漏  3.数据逻辑上的先后关系，有的数据只能在特定情况下才有  
```python
# ---- drop ----
data.isnull.sum() / len(data)  # 缺失比例
data.dropna(axis= )
data.drop([])

# ---- imputation ----
from sklearn.impute import SimpleImputer
# SimpleImputer is based on mean value imputation

imputer = SimpleImputer()
data_impute = imputer.fit_transform(data)  # 'transform' for valid/test
pd.DataFrame(data_impute, index= , columns= )
...

# ---- imputation extension ----
# 数据项缺失本身可能就包含了一些信息，有时候是对预测有用的，所以在补全之外加上某特征是否有缺失的表示
data[col+'_bool'] = data[col].isnull()
imputation...
```
**Categorical Variables**  
```python
# ---- label encoding ----
# 序列型类别ordinal variables，用有序数值打标签可以反应出它们的先后顺序，在诸如决策树模型中很有用
from sklearn.preprocessing import LabelEncoder
...

# ---- one-hot encoding ----
# 类别之间没有先后顺序，nominal variables
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unkown='ignore', sparse=False)
data_encode = encoder.fit_transform(data)  # 'transform' for valid/test
data_encode = pd.DataFrame(data_encode, index= , columns= )
...

# ---- count encoding ----
# 用类别出现的次数作为编码，注意只能使用training data统计!!!
from category_encoders import CountEncoding
...

# ---- target encoding ----
# 各类别下target标签的均值来编码
from category_encoders import TargetEncoding
...

# ---- catboost encoding ----
#  同样面向target, 只不过是在当前数据之前的类别下target的均值
from category_encoders import CatBoostEncoding
...
```
**Numerical Variables**  
主要是数值分布不均的问题，常见长尾分布，用pandas绘制柱状图观察，数据取平方根（决策树有效）或者对数（使数值接近高斯分布，对深度学习模型有效）  

> 下列相关的操作都是基于在baseline model上的，比如Lasso regression, decision tree, random forests等，使用这些模型得到的基础训练结果可以为进一步的feature engineering提供信息，主要关注特征选择，找出重要特征以后还可以对这些特征进行进一步的特征组合  

**Multual Information**  
各特征与target的互信息  
```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, K=10)
data_new = selector.fit_transform(data, target)  # 选取出互信息最大的K个特征
data_new

# 为了便于查看原始数据的筛选结果，需要对data_new逆变换
selected_data = pd.DataFrame(selector.inverse_transform(data_new), 
                                 index=data.index, 
                                 columns=data.columns)
selected_data.head()
```
**L1 Regression: Lasso**  
L1范数惩罚项进行的回归得到的结果为稀疏矩阵，主要用这个特性进行特征的筛选，惩罚系数越大，保留的特征越少，它与L2范数惩罚项不同(Ridge回归用于避免overfitting)  
最优的惩罚项系数还是要通过验证集来选择  
```python
# from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

train, valid, _ = train_test_split(data)
X, y = train[features], train[target]

# Set the regularization parameter C=1
logistic = LogisticRegression(C=1, penalty="l1", solver='liblinear', random_state=1).fit(X, y)
model = SelectFromModel(logistic, prefit=True)

X_new = model.transform(X)
X_new

# transform back
selected_features = pd.DataFrame(model.inverse_transform(X_new), 
                                 index=X.index,
                                 columns=X.columns)
```
**Permutation Importance**  
以验证数据集为参考，将训练好的模型应用在某一列特征随机乱序后的数据上，观察预测效果的改变
`fast to compute; widely used, easy to understand; consistent`  
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

train_X, valid_X, train_y, valid_y = train_test_split(X, y, random_state=1)
model = RandomForestClassifier(n_estimators=100,
                                  random_state=0).fit(train_X, train_y)

***********
import eli5
***********
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
```
**Partial Plots**  

  Machine Learning Explainability
  - debugging
  - informing feature engineering
  - directing future data collection
  - informing human decision
  - build trust
