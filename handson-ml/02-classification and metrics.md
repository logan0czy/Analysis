# Performance Measures
For binary classification
- [Accuracy](#acc)
- [Confusion Matrix](#conf-matrix)
- [Precision and Recall](#pr)
- [PR Trade-off](#pr-tradeoff)
- [ROC curve](#roc)

<span id='acc'></span>
## Accuracy
正负样本分类的正确率。验证accuracy时通过k折交叉验证的方法得到在每一个训练样本上的`clean prediction`。自定义的分层抽样k折验证法可以通过下述得到
```python
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_split=3)
for train_id, test_id in skfolds.split(X_train, based_target):
    clf_clone = clone(clf)
    [...] # fit, predict, get scores
```
但是当样本分布出现数据倾斜时（`skewed datasets`），正确率常常反应不了模型的优劣，例如正确率较高的模型其实什么事也没做，只是把样本划分为占比高的那一类。

<span id='conf-matrix'></span>
## Confusion Matrix
查看每一类别被正确或者错误划分的样本数量。此时需要使用`cross_val_predict`查看模型对每一个训练样本的预测类别。
```python
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3)

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_train, y_train_pred)
conf_matrix
```
通常情况下，计算混淆矩阵的目的是使用可视化的方法来对模型预测进行错误分析，这在[错误分析](#err-analys)一节作了详尽描述。

<span id='pr'></span>
## Precision and Recall
精确度（precision）和召回率（recall, sensitivity, true positive rate-TPR）是从混淆矩阵中得来的更为简洁的评估指标，其中，`precision = TP/(TP+FP)`；`recall = TP/(TP+FN)`。此外，也能用F1值来综合两者的大小，即它们的调和平均数`F1 = 2 x precision x recal / (precision + recall)`，F1值只有在两者都比较大时才能取得一个较大的值。
```python
from sklearn.metrics import precision_score, recall_score, f1_score
```
`The F1 score favors classifiers that have similar precision and recall`。实际应用中我们常常需要在两者之间进行折衷考量，书中举了两个有意思的例子：对影片是否适合儿童观看进行分级，这就需要有很高的精确度，展示给儿童的影片不能出现任何不适宜的内容，不可避免得这种分级会错分一些优秀的儿童影片；商店盗窃者筛选，这就需要“宁可错杀，不可放过”的原则，需要分类器有较高的召回，之后再进行人工筛选。

<span id='pr-tradeoff'></span>
## PR Trade-off
scikit-learn中的分类器的分类判决准则是固定的，比如线性分类器sgd_clf,logistic_clf等是根据0分界线，决策树根据0.5概率等。但有时候根据实际需要，常常需要根据分类器的输出值（通过`decision_function`,`predict_proba`获得），手动定义判决门限来进行分类，方法就是绘制prediction-recall-curve。
```python
y_scores = cross_val_predict(sgd_clf, X_train, y_train, cv=3, method='decision_function')

from sklearn.metrics import precision_recall_curve
precisions, recalls, threshholds = precision_recall_curve(y_train, y_scores)
[...] # plot precision/recall versus threshold
[...] # plot precision versus recall

thresholds[np.argmax(precisions >= value_needed)] # get corresponding threshold for a specific precision
```
`If someone says "Let's reach 99% precision," you should ask, "At what recall?"`

<span id='roc'></span>
## ROC Curve
ROC(receiver operating characteristic), 绘制的是true positive rate-TPR与false positive rate-FPR之间的曲线，曲线越靠近左上角，分类器越好，在0.5对角线的是随即分类器。AUC(area under the curve), 计算的是ROC曲线下的面积，AUC值越大，表明分类器越好。
```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train, y_scores)
[...] # plot the curve

from sklearn.metrics import roc_auc_score
roc_auc_score(y_train, y_scores)
```
> Tip
>> As a rule of thumb, you should prefer the PR curve whenever the positive class is rare or when you care more about the false positives than the false negatives. Otherwise, use the ROC curve.

ROC曲线可以在初始探索的时候使用，后续如果要进一步提升指标，数据集又出现了倾斜的情况的话，就需要PR曲线来更精确地调整策略了。

# Classification Categories
## Binary Classification
基本的二元分类问题

## Multiclass Classification
多元分类。 SGD, Random Forest, naive Bayes能够直接处理多元分类问题，Logistic Regression, SVM是二元分类器，需要用一定的策略使得它们也能处理多元分类的问题。
- One versus One (OvO) `see which class wins the most duels.` 主要的优点是每一个对应的分类器只需要对数据集的一部分进行训练，减少了数据处理量。比如SVM `scale poorly with the size of the training set`。
- One versus the Rest (OvR) / One versus All (OvA) 大部分情况下使用

使用特定的策略通过如下代码
```python
from sklearn.multiclass import OneVsRestClassifier
ovr_clf = OneVsRestClassifier(some_clf_class())
[...]
```

## Multilabel Classification
多标签分类。相比于多元分类，多标签分类的输出不一定仅有一个类别为1，比如监控画面中人脸识别图像中有哪些人。方法有`sklearn.neighbors.KNeighborsClassifier`。评估中添加对应参数`e.g., f1_score(y_multilabel, y_train_pred, average='macro')`

## Multioutput Classification
多输出分类。每一个输出值不再是0-1二元值，可以是多值。应用比如图像去噪。

<span id='err-analys'></span>
# Error Analysis
经历了基本的数据处理，模型选择和调整的过程之后，进一步提升模型性能就要进行更精细的错误分析了。首选混淆矩阵，其次就是对每一个特定的实例分析。
```python
# raw counts
plt.matshow(conf_mx, cmap=plt.cm.hot)

# error ratio
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)

plt.matshow(norm_conf_mx, cmap=plt.cm.hot)
```
之后就是进行数据处理，创造新特征，特征组合等等
