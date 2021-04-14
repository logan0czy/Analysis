# Decision Trees
Scikit-Learn决策树的实现主要是基于CART算法。不同于其它模型，决策树对数据处理没有太多要求（如特征放缩、归一化等）。

## Visualization
First output a graph definition ".dot" file.  
```python
from sklearn.tree import export_graphviz
export_graphviz(
  tree_clf,
  out_file='image_path',
  feature_names=dataset.feature_names,
  class_names=dataset.target_names,
  rounded=True,
  filled=True
)
```
Then you can use the dot command-line tool from the Graphviz package to convert this .dot file to a variety of formats, such as PDF or PNG. Graphviz is an open source graph visualization software package, available at http://www.graphviz.org/.
```
dot -Tpng iris_tree.dot -o iris_tree.png
```

## Impurity: Gini or Entropy?  
一般来讲，没有太大的区别。Gini指数相对的计算消耗更小，所以是不错的默认选择。可能的不同只有Gini指数倾向于为出现频率较高的类单独开辟一个分支，而基于熵的话生成的决策树更平衡。

## Instability

- Love orthogonal decision boundaries, which makes them sensitive to training rotation.(fix: PCA)  
- Sensitive to small variations.  
- Feature selective is stochastic.  

Fix: Random Forest, by averaging predictions over many trees.
