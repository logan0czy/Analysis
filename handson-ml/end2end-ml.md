# End-End Machine Learning Project Process

业务上每个人做的都会是整个机器学习落地过程（Pipeline）的中间一环，因此明白自己当前任务在pipeline中处于的角色很重要。这包括：  
明确任务需求，做这个的目的是什么，是服务于下游的机器学习模型，还是做出确定的决策  
上游或者说从data store中取出的数据有没有进行过预处理或者其他的含义  
下游任务对当前任务的需求，例如是需要精确的预测值/类别型的预测分类  

合适的评价指标，RMSE/MAE... L-p norm p值越大对数值大的项敏感性更高，当然会越来越忽略数值小的项。  

Sampling bias:  
train-test split的时候，random split可能造成training set和test set数据分布不同。这就需要`Stratified Sampling`，根据对预测值最重要的特征项进行数据分桶，按类别比例进行数据采样分割。
