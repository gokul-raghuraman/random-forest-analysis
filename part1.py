"""
Decision Tree with no attributes removed
"""
from decision_tree import *
from cross_validation import CrossValidation

def runExperiment():
    with open("data_banknote_authentication.txt", "r") as dataset:
        bankNoteData = np.asarray([line.split(",") for line in dataset.readlines()], dtype=np.float64)
        
    numFolds = 10
    crossVal = CrossValidation(bankNoteData, numFolds)
    
    for fold in range(crossVal.folds):
        print("\nFOLD : " + str(fold + 1))
        
        trainingData = crossVal.getTrainingData(fold)
        attrIndices = [i for i in range(bankNoteData.shape[1] - 1)]
        decisionTree = DecisionTree(bankNoteData, trainingData, attrIndices)
        makeDecisionTree(bankNoteData, decisionTree.root, decisionTree.root.allowedAttrs, None)
        
        testingData = crossVal.getTestingData(fold)
        outputs = [int(bankNoteData[item, 4]) for item in testingData]
        crossVal.addOutputs(outputs)
        
        queries = [bankNoteData[item, :4] for item in testingData]
        predictedOutputs = [classifyQuery(query, decisionTree.root) for query in queries]
        crossVal.addPredictedOutputs(predictedOutputs)
        
        print("OUTPUTS = " + str(outputs))
        print("PREDICTED OUTPUTS = " + str(predictedOutputs))
        
    crossVal.printConfusionMatrix()
    
    print("ACCURACY  = " + str(crossVal.getAverageAccuracy() * 100) + "%")
    print("PRECISION = " + str(crossVal.getAveragePrecision() * 100) + "%")
    print("RECALL    = " + str(crossVal.getAverageRecall() * 100) + "%")
    

if __name__ == '__main__':
    runExperiment()