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
from sklearn.pipeline import Pipeline
from sklean.compose import ColumnTransformer
full_pipe = ColumnTransformer([
        ('pipe1', transformer1, attr1),
        ('pipe2', transformer2, attr2),
    ])
```
> Instead of using a transformer, you can specify the string "drop" if you want the columns to be dropped, or you can specify "passthrough" if you want the columns to be left untouched. By default, the remaining columns (i.e., the ones that were not listed) will be dropped, but you can set the remainder hyperparameter to any transformer (or to "passthrough") if you want these columns to be handled differently.

<span id='title5'></span>
## [Select a model and train it](#head)
**K-fold cross-validation**, `sklearn.model_selection.cross_val_score`. 在深入某一个模型之前，多尝试几个模型看看预测的效果，这能得到一些基本的比较和参考，得到模型的备选集。  
> You should save every model you experiment with so that you can come back easily to any model you want. Make sure you save both **the hyperparameters and the trained parameters, as well as the cross-validation scores and perhaps the actual predictions as well**. This will allow you to easily compare scores across model types, and compare the types of errors they make. You can easily save Scikit-Learn models by using Python’s pickle module or by using the **joblib** library, which is more efficient at serializing large NumPy arrays (you can install this library using pip):
```
import joblib
joblib.dump(my_model, "my_model.pkl")
# and later...
my_model_loaded = joblib.load("my_model.pkl")
```

<span id='title6'></span>
## [Fine-tune your model](#head)
*Grid Search*  
```
from sklearn.model_selection import GridSearchCV
param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]}, 
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3,4]}, 
    ]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
```
*Randomized Search*  
当超参数组合数过多时，偏向于使用随机搜索的方法检索超参数的可能值。 

**Analyze the Best Model and Their Errors**  
分析得到的最好模型中各个属性特征的重要性程度，以及预测出错的部分可能的原因。  
将测试集/验证集中的数据分割为不同类型的子集，观察模型在各个自己上的表现是分析模型出错原因的不错思路  
**Evaluate Your System on the Test Set**  
compute a confidence interval for the generalization error.  
```
from scipy import stats
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, 
                         loc=squared_errors.mean(), scale=stats.sem(squared_errors)))
```

<span id='title7'></span>
## [Present your solution](#head)
1.记录重要的结果发现。2.清晰简明的图表表示。3.简洁的基本结论。  
其中要记录的包括：学习到什么；有效的和无效的部分；模型中的假设；模型or系统的局限性。。。  

<span id='title8'></span>
## [Launch, monitor, and maintain your system](#head)  
数据分析中模型训练反而是耗时最少的一环，更加需要长久处理的是数据预处理以及模型发布后的更新维护环节。  
模型部署后需要编写监控脚本随时监控模型在线预测的表现，模型效果有可能出现急剧下降，也有可能随着时间的推移缓慢变化（输入输出数据概率分布随着时间有所变化）。监控模型性能可以通过下游的任务的提升效果反映；或者通过抽样进行人工的结果分析。  
自动化更新在线模型的一些tips:  
- 持续收集新数据  
- 训练及fine-tune模型的自动化脚本  
- 比较旧模型及新模型在新的数据集上性能表现的脚本，以此决定是否发布新版模型  

此外，也需要监控输入数据的质量（分布不均，缺失值过多，出现新的数据类型等等）。  
**确保备份各版本数据集以及各版本的模型，以方便当前模型效果变差时的回退**
