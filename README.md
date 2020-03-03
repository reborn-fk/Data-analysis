# Data-analysis
daily notes
数据特征选择：
在建立一个机器学习模型时，并不是所有所有的数据属性都对模型有同等的贡献，因此也不是数据属性越多越好。在建立模型之前，要从众多的数据属性中选择对模型的输出和结果预测贡献最大的那些变量，这种对特征进行筛选的过程叫做“特征选择”。“特征选择”具有以下三方面的重要作用：

1. 减少过度预测(overfit)，减少噪音变量对模型的贡献。
2. 增加准确性，依靠减少无关的变量提高模型的预测准确性。
3. 减少模型训练时间，越少的训练数据，意味着越少的训练时间。


常见的模型特征选择有两种：（scikit-learn中有两种特征选择的方法，一种叫做循环特征消减(Recursive Feature Elimination)和特征重要性评级 (feature importance ranking)）

1. 循环特征消减(Recursive Feature Elimination)
作为一种特征选择方法，其工作原理是：循环地移除变量和建立模型，通过模型的准确率来评估变量对模型的贡献。以下代码使用UCI的Iris数据集，使用sklearn.feature_selection的RFE方法来实现该方法。
      from sklearn import datasets
      from sklearn.feature_selection import RFE
      from sklearn.linear_model import LogisticRegression
      dataset =datasets.load_iris() # laod iris dataset
      model = LogisticRegression() # build logistic regression model
      rfe = RFE(model,3) # limit number of variables to three
      rfe = rfe.fit(dataset.data,dataset.target)
      print(rfe.support_) 
      print(rfe.ranking_)

2. 特征重要性评级 (feature importance ranking)
“组合决策树算法”（例如Random Forest or Extra Trees）可以计算每一个属性的重要性。重要性的值可以帮助我们选择出重要的特征。以下代码使用UCI的Iris数据集，使用sklearn.metrics和sklearn.ensemble 的ExtraTreesClassifier来实现该算法。

      from sklearn import datasets
      from sklearn import metrics
      from sklearn.ensemble import ExtraTreesClassifier
      dataset =datasets.load_iris() # laod iris dataset
      model = ExtraTreesClassifier() # build extra tree model
      model.fit(dataset.data,dataset.target)
      print(model.feature_importances_) #display importance of each variables

