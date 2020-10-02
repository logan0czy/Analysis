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
编写一个自动化的脚本获取数据集，对数据集的大致内容，数据类型组成，简单统计信息，简单的可视化图像有一个基本的了解。  
> **train-test split**:  
训练集测试集拆分，NEVER EVER LOOK AT THE TEST SET。一般按8:2的比例分割出测试集，并且将测试集对应的的数据进行固定（固定方法可以根据每个实例中unique的属性对应的哈希值来划分数据集）。  
>> **sampling bias**: 由于随机抽样导致训练集和测试集得到的数据分布不一致（因为机器学习本质上是对同一概率分布下的观测数据进行拟合预测）。解决办法是分层抽样（Stratified Sampling)，scikit-learn有现成的工具，根据对预测值最重要的那个属性的分布进行分层抽样（可能需要有数据分桶的操作）。  

<span id='title3'></span>
## [Discover and visualize the data to gain insights](#head)

<span id='title4'></span>
## [Prepare the data for Machine Learning algorithms](#head)

<span id='title5'></span>
## [Select a model and train it](#head)

<span id='title6'></span>
## [Fine-tune your model](#head)

<span id='title7'></span>
## [Present your solution](#head)

<span id='title8'></span>
## [Launch, monitor, and maintain your system](#head)    
