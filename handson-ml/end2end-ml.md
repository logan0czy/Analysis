<span id='head'></span>
# End-End Machine Learning Project Process  
- [Look at the big picture](#title1)  
- [Get the data](#title2)
- [Discover and visualize the data to gain insights](#title3)  
- [Prepare the data for Machine Learning algorithms](#title4)  
- [Select a model and train it](#title5)  
- [Fine-tune your model](#title6)  
- [Present your solution](#title7)  
- [Launch, monitor, and maintain your system](#title8)  

<span id='title1'></span>
## [Look at the big picture](#head)
- Frame the problem: 业务上每个人做的都会是整个机器学习落地过程（Pipeline，各组件的接口是处理产生的数据）的中间一环，因此明白自己当前任务在pipeline中处于的角色很重要。弄清当前所做的模型怎样使用或者说团队如何从中获得效益/已有的参考模型benchmark  
- Performance measure  
- Check the assumptions: 进一步明白模型的需求，对模型的一些业务上的假设，如下游所需的输入结果是预测的类别。  

<span id='title2'></span>
## [Get the data](#head)
编写一个自动化的脚本获取数据集，对数据集的大致内容，数据类型组成，简单统计信息，简单的可视化数值分布图像有一个基本的了解。  
> **train-test split**:  
训练集测试集拆分，NEVER EVER LOOK AT THE TEST SET。一般按8:2的比例分割出测试集，并且将测试集对应的的数据进行固定（固定方法可以根据每个实例中unique的属性对应的哈希值来划分数据集）。  
>> **sampling bias**: 由于随机抽样导致训练集和测试集得到的数据分布不一致（因为机器学习本质上是对同一概率分布下的观测数据进行拟合预测）。解决办法是分层抽样（Stratified Sampling)，scikit-learn有现成的工具，根据对预测值最重要的那个属性的分布进行分层抽样（可能需要有数据分桶的操作）。  

<span id='title3'></span>
## [Discover and visualize the data to gain insights](#head)
- Visualizing data  
- Looking for correlations  
- Experiment with attribute combinations  
### Visualizing data
如果训练集整个过大，可以选取一部分进行可视化
```
pd.DataFrame.plot(kind='scatter', x='attr1', y='attr2', alpha=0.4,
    s='attr3', label='label', figsize=(20, 18),
    c='attr4', cmap=plt.get_cmap('jet'), colorbar=True,
)
```

### Looking for correlations
各个属性之间的*线性相关性*  
```
import pandas as pd
corr_matrix = pd.DataFrame.corr()
corr_matrix['target_attr'].sort_values(ascending=True)

from pandas.plotting import scatter_matrix
attributes = ['attr1', 'attr2', 'attr3', 'attr4', 'attr5']
scatter_matrix(pd.DataFrame[attributes], figsize=(12, 8))
```

### Experiment with attribute combinations
根据预测目标以及实例拥有的各种属性猜测有用的属性之间的组合关系，从而为实例添加新的属性来帮助预测的准确性。另外在探索阶段不需要穷尽每一种可能的属性组合，只需要产生几种就可，属性的其他可能组合可以在之后根据训练结果再进行多轮的修改。  

<span id='title4'></span>
## [Prepare the data for Machine Learning algorithms](#head)
**Write functions, form a pipeline**  
- [Data cleaning](#title4-1)  
- [Handling text and categorical attributes](#title4-2)  
- [Custom transformers](#title4-3)  
- [Feature scaling](#title4-4)  
- [Transformation pipelines](#title4-5)  
> **Scikit-learn API design**  
> Consistency:  
>> Estimators: -> `fit()`method  
估计数据集中的相关统计信息。  
Transformers: -> `transform()`method  
根据estimator中学习到的参数对数据集进行变化。一般`fit_transform()`底层优化的速度更快。  
Predictors: -> `predict(), score()`method

> Inspection  
>> All the estimator’s hyperparameters are accessible directly via public instance variables (e.g., imputer.strategy), and all the estimator’s learned parameters are accessible via public instance variables with an underscore suffix (e.g., imputer.statistics_).  

> Nonproliferation of classes  
>> Datasets are represented as NumPy arrays or SciPy sparse matrices, instead of homemade classes. Hyperparameters are just regular Python strings or numbers.  

> Composition  
>> Existing building blocks are reused as much as possible. For example, it is easy to create a Pipeline estimator from an arbitrary sequence of transformers followed by a final estimator, as we will see.  

> Sensible defaults  
>> Scikit-Learn provides reasonable default values for most parameters, making it easy to quickly create a baseline working system.  

<span id='title4-1'></span>
### Data cleaning
数值型数据项的缺失数据删除或插值补全，因为不确定各个属性之后的新实例中会不会遇到缺失值，所以最好对每一个数值属性进行插值的相关统计量的计算。
```
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
imputer.fit(pd.DataFrame)
imputer.transform(pd.DataFrame)
```

<span id='title4-2'></span>
### Handling text and categorical attributes
转化为表示类别的数值，若类别之间有顺序性，使用sklearn.preprocessing.OrdinalEncoder；若类别之间是没有任何相关性的，使用sklearn.preprocessing.OneHotEncoder，（Scipy存储产生的稀疏矩阵）。  
当整个类别的数量过多后也会影响模型的学习效果，其中一个办法是用一些相关的数值来进行表示和替代；另一个办法就是使用Embedding等的技术。  

<span id='title4-3'></span>
### Custom transformers
因为Scikit-Learn依照的是`duck typing`，所以可以自定义transformer（`fit()->返回self, transform(), fit_transform()`）。  
> You can get the last one for free by simply adding **TransformerMixin** as a base class. If you add **BaseEstimator** as a base class (and avoid `*args` and `**kargs` in your constructor), you will also get two extra methods (get_params() and set_params()) that will be useful for automatic hyperparameter tuning.

<span id='title4-4'></span>
### Feature scaling
**min-max scaling**: also called normalization, `MinMaxScaler` from Scikit-Learn.  
**standardization**: `StandardScaler` from Scikit-Learn.  
> Unlike min-max scaling, standardization does not bound values to a specific range, which may be a problem for some algorithms (e.g., neural networks often expect an input value ranging from 0 to 1). However, standardization is much less affected by outliers.

<span id='title4-5'></span>
### Transformation pipelines
除了最后一个以外其他的元素都需要是transformer，pipeline使用的是与最后一个estimator相同的API。
```

<span id='title5'></span>
## [Select a model and train it](#head)

<span id='title6'></span>
## [Fine-tune your model](#head)

<span id='title7'></span>
## [Present your solution](#head)

<span id='title8'></span>
## [Launch, monitor, and maintain your system](#head)    
