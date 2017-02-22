from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext

# Create the spark context and set logging level
sc = SparkContext()
sc.setLogLevel('WARN')

# Load and parse the data file into an RDD
data = MLUtils.loadLibSVMFile(sc, '/user/cloudera/iris.libsvm')
# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a DecisionTree model
modelDT = DecisionTree.trainClassifier(trainingData, numClasses=3, categoricalFeaturesInfo={}, impurity='gini', maxDepth=5, maxBins=32)

# Evaluate model on test instances and compute test error
predictions = modelDT.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testAccuracy = labelsAndPredictions.filter(lambda (v, p): v == p).count() / float(testData.count())
print('Decision Tree Test Accuracy =' + str(testAccuracy))

# Print the predictions
print ("Acutual vs Predicted values - Decision Tree")
print (labelsAndPredictions.collect())


# Train a RandomForest model
modelRF = RandomForest.trainClassifier(trainingData, numClasses=3, categoricalFeaturesInfo={},numTrees=3, featureSubsetStrategy="auto", impurity='gini', maxDepth=4, maxBins=32)

# Evaluate model on test instances and compute test error
predictions = modelRF.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testAccuracy = labelsAndPredictions.filter(lambda (v, p): v == p).count() / float(testData.count())
print('Random Forest Test Accuracy =' + str(testAccuracy))

# Print the predictions
print ("Acutual vs Predicted values - Random Forest")
print (labelsAndPredictions.collect())

