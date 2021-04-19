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
### Bagging
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

<span id='boosting'></span>
### Boosting

<span id='stacking'></span>
### Stacking

<span id='rf'></span>
## [Random Forests](#head)
