<span id='head'></span>
# Ensemble Learning - wisdom of the crowd
- [Voting Classifiers](#voting)
- Ensemble Methods
  - [Bagging](#bagging)
  - [Boosting](#boosting)
  - [Stacking](#stacking)
- [Random Forests](#rf)  

**强可学习与弱可学习的等价性：一个强学习器可以通过综合多个弱学习器的结果来得到**。文中举了典型的掷偏心硬币的例子：一枚单次掷正反面概率分别为0.51和0.49的硬币（对应到一些分类器只比随机猜测好那么一点），很多次投掷，正面朝上占多数的概率会接近75%！由大数定律也可以看出，当投掷足够多次时，正面朝上的比例会基本稳定在51%左右。那么对应到分类器，如果使用的分类器足够多，最终判决时依据多数表决的方法得到正确结果的概率会明显高出单个学习器的判决正确率。这就是集成学习的理论依据。

<span id='voting'></span>
## [Voting Classifiers](#head)  
- **hard voting**: 多数表决的规则。根据每个分类器的判别结果，选择占多数的类别为最终判定。  
- **soft voting**: 对类别估计的概率的平均。这种方法会给置信度高的结果以更多的权重，因此相对于hard voting会得到更好的结果。（不过在scikit-learn的实现中需要每一个基本分类器都能估计类别的概率，即有predict_proba函数）  

## [Ensemble Methods](#head)  
集成学习的一个大前提就是每一个基分类器一定要是独立的，它们所犯的错误相互之间也没有关联性，否则用再多数量的分类器对结果的提升也没有帮助，这就需要增大每一个基分类器的**diversity**。一种方法是使用差异极大的机器学习方法，另一种方法便是在训练所基于的数据集上下手（毕竟机器学习的本质是学习服从一定概率分布的数据所包含的规律）。

<span id='bagging'></span>
### Bagging - 改变选取的样本集
- Bootstrap aggregating:用于每一个基分类器的训练样本都是从总样本中通过有放回的随机抽样获得。这样每个分类器相对于直接使用总样本的分类器有更高的bias和variance，但是通过aggregating，bias差距不大，variance能够得到极大降低。  
- Pasting: 无放回的抽样。bagging由于在基分类器对应的训练数据中可能对同一个样本重复抽取，因而相对于pasting的多样性更高，训练效果也相对更好。  
```python
from sklearn.ensemble import BaggingClassifier
# automatically perform soft voting
```
因为bagging方法中各基分类器都是基于独立抽样形成的数据集进行独立地训练，因而天然的有很好的**并行性**和**可扩展性**。  

#### Out-of-Bag Evaluation  
对任意一个分类器来说，有的样本可能被抽取多次，而有的可能一次也没被选中，对训练样本总量为m且随机抽样总数为m的bagging来说，每一个predictor使用到的样本量约占总量的63%，因而还有37%的样本完全没有被使用过！这些样本天然的可以被用作验证集用于评估每一个基分类器的性能。  
> You can evaluate the ensemble itself by averaging out the oob evaluations of each predictor. In Scikit-Learn, you can set **oob_score=True** when creating a **BaggingClassifier** to request an **automatic oob evaluation** after training. The resulting evaluation score is available through the **oob_score_ variable**.  

#### Random Patches and Random Subspaces  
- Random Patches: 对样本和特征的随机抽样。This technique is particularly useful when you are dealing with high-dimensional inputs (such as images)。  
- Random Subspaces: 仅对特征的随机抽样。  

<span id='boosting'></span>
### Boosting - 改变各样本的权重  
依次地训练各分类器/回归器，序列中每一个predictor在训练时目标都是修正predecessor的错误。因而不同于bagging方法，提升方法boosting没有办法并行地生成各基分类器。    

#### AdaBoost  
每一个基分类器在训练时，依照前一个基分类器的分类准确率，相应地改变训练样本的权重，即增大分类错误样本的权重，重点对这些样本进行学习。  
> As you can see, this sequential learning technique has some similarities with Gradient Descent, except that instead of tweaking a single predictor’s parameters to minimize a cost function, AdaBoost adds predictors to the ensemble, gradually making it better.  
> Once all predictors are trained, the ensemble makes predictions very much like bagging or pasting, except that predictors have different weights depending on their overall accuracy on the weighted training set.  

Steps:
1. weighted error rate of j-th predictor  
2. predictor weight: logrithm with learning rate expression. `The more accurate the predictor is, the higher its weight will be. If it is just guessing randomly, then its weight will be close to zero. However, if it is most often wrong (i.e., less accurate than random guessing), then its weight will be negative`  
3. update the instance weights: exponential boosting and then normalize.  

Scikit-Learn uses a multiclass version of AdaBoost called SAMME, if the predictors can estimate class probabilities, a variant of SAMME called SAMME.R can be used.  
解决过拟合的问题，通过减少estimator的数量或者对基分类器进行regularization。

#### Gradient Boosting  
新分类器是对前序估计器的残差进行拟合，最终结果是将各估计器的结果相加得到的。  
**shrinkage**:一种正则方法，用learning rate代表每一个树的贡献程度，学习率越小，所需的树越多，但相应的准确率会越高。  
**early stopping**-防止过拟合的方法：  
- staged_predict():模型训练完毕后查看每一阶段模型在验证集上的表现从而确定最优估计器数量。  
- warm_start=True:每一阶段训练完毕后在验证集上看模型表现，当损失函数没有进一步降低时则终止boosting。  

**Stochastic Gradient Boosting**: 每一个基分类器在训练时的样本都是随机抽样得到的，因而训练速度会相对更快一些。  

Gradient Boosting推荐的实现：[XGBoost](https://github.com/dmlc/xgboost)

<span id='stacking'></span>
### Stacking
相比于用确定的函数对各基分类器的结果进行aggregating，stacking方法训练了单独的一个predictor来基于各基分类的结果进行判决，即加了一层分类器，并且可以照此方法层层堆叠上去（类似于神经网络）。每一层分类器的对应一个独立训练集，并在基于上一层训练好的参数来进行训练。  

<span id='rf'></span>
## [Random Forests](#head)
采用bagging的方法，base estimator是决策树的一种很有效的集成学习方法。  
为了进一步加强各决策树的diversity，随机森林的每一个决策树在生成的时候，各节点选择的最优分割特征是在总特征集中的一个随机子集上进行选取的（random split）。  
**Extra-Trees**: extremely randomized trees，进一步地增加各决策树生成的随机性，各节点选择最优特征的分割门限的时候也是随机选择的。  
`It's hard to tell whether random forests or extra-trees is better`  
### Feature Importance  
随机森里天然的具有特征选择的可解释性。在最终生成的随机森林中，计算各特征被用来分割节点的加权平均个数，就能清楚的知道哪些特征对于降低分类的impurity更加的重要。而加权平均的计算方法是依据各节点所对应的训练样本数量来计算的。
（通过`feature_importance_`变量来得到）  
