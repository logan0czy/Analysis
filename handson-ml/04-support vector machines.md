# Support Vector Machines  

## Classification  
> Be well suited for classification of complex small- or medium-sized datasets.  

基于`sklearn.svm`中不同实现下的svm方法, (all based on 'hinge' loss).    
|  | base | time complexity | scaling required | kernel trick | note |
|:----:|:----:|:----:|:----:|:----:|:----:|
| LinearSVC | liblinear | O(mxn) | yes | no | regularize the bias, set param `dual`=False |
| SVC | libsvm | O(m2xn) to O(m3xn) | yes | yes | scales well with sparse features |
|SGDClassifier | | O(mxn) | yes | no | |

常用的kernel: 'linear', 'poly', 'rbf'

## Regression  
> To use SVMs for regression instead of classification, the trick is to reverse the objective: instead of trying to fit the largest possible street between two classes while limiting margin violations, SVM Regression tries to fit as many instances as possible **on** the street while limiting margin violations (i.e., instances **off** the street).  

Methods: 'LinearSVR', 'SVR'.  

## Online SVMs  
基于正则化的合页损失函数(hands-on-ml 中称为 hinge loss)，使用SGDClassifier进行求解。  
（由于合页损失函数在分割点不可导，使用的是subderivative的方式，将该点导数设为左右两条线之间导数中间的任意值）
