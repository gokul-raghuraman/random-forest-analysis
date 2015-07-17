import numpy as np
import math
import random


def removeAttrsRandom(inputData):
    modifiedData = []
    
    shuffledRowIndices = [i for i in range(inputData.shape[0])]
    np.random.shuffle(shuffledRowIndices)
    shuffledRowsParts = np.asarray(np.array_split(shuffledRowIndices, 10))
    
    for i in range (shuffledRowsParts.shape[0] - 1):
        for index in shuffledRowsParts[i]:
            modifiedData.append(inputData[index])
    
    for example in inputData[shuffledRowsParts[shuffledRowsParts.shape[0] - 1], :]:
        attrIndexToRemove = random.randint(0, 3)
        example[attrIndexToRemove] = -20
        modifiedData.append(example)
    
    np.random.shuffle(modifiedData)
    return np.asarray(modifiedData)


class DecisionTree:
    def __init__(self, dataSet, examples, attributes):
        self.root = Node(dataSet, examples, attributes, None)


class Node:
    leftChild = None
    rightChild = None
    
    def __init__(self, dataSet, examples, allowedAttrs, leafClass):
        self.dataSet = dataSet
        self.examples = np.asarray(examples)
        self.allowedAttrs = allowedAttrs
        self.leafClass = leafClass
        
        if (self.leafClass is None):
            classCount = np.bincount(np.asarray(self.dataSet[examples, 4], dtype="int"))
            
            p = self._getSafeValue(classCount, 1)
            n = self._getSafeValue(classCount, 0)
            
            self.entropy = self.getEntropy(p, n)

            (self.bestAttr, self.bestTheta) = self.getBestAttr()
               

    def getBestAttr (self):
        totalNum = float(self.examples.shape[0])
        allInfoGains = []
    
        for attrIndex in self.allowedAttrs:
            bestTheta = self.getBestTheta (attrIndex)
            leftExamples = [index for index in self.examples if self.dataSet[index][attrIndex] != -20 and 
                            self.dataSet[index][attrIndex] <= bestTheta]
            rightExamples = [index for index in self.examples if self.dataSet[index][attrIndex] != -20 and 
                             index not in leftExamples]
            uncertainExamples = [index for index in self.examples if self.dataSet[index][attrIndex]== -20]
            
            
            if (len(leftExamples) > len(rightExamples)):
                leftExamples += uncertainExamples
            else:
                rightExamples += uncertainExamples
            
            leftExamples = np.asarray(leftExamples)
            rightExamples = np.asarray(rightExamples)
        
            numLeft = float(leftExamples.shape[0])
            numRight = float(rightExamples.shape[0])
        
            if not leftExamples.tolist():
                numClassesLeft = [0, 0]
            else:
                numClassesLeft = np.bincount(np.asarray(self.dataSet[leftExamples, 4], dtype="int"))
            if not rightExamples.tolist():
                numClassesRight = [0, 0]
            else:
                numClassesRight = np.bincount(np.asarray(self.dataSet[rightExamples, 4], dtype="int"))
        
            pLeft = self._getSafeValue(numClassesLeft, 1)
            nLeft = self._getSafeValue(numClassesLeft, 0)
            pRight = self._getSafeValue(numClassesRight, 1)
            nRight = self._getSafeValue(numClassesRight, 0)
        
            entropyLeft = self.getEntropy (pLeft, nLeft)
            entropyRight = self.getEntropy (pRight, nRight)
            expectedEntropy = (numLeft / totalNum) * entropyLeft + (numRight / totalNum) * entropyRight
            allInfoGains.append([self.entropy - expectedEntropy, bestTheta])
        
        bestAttr = self.allowedAttrs[allInfoGains.index(max(allInfoGains))]
        bestTheta = allInfoGains[allInfoGains.index(max(allInfoGains))][1]
        return [bestAttr,  bestTheta]
    
    @staticmethod
    def _getSafeValue(array, index):
        try:
            val = float(array[index])
        except IndexError:
            val = 0.0
        return val  


    def getBestTheta (self, attrIndex):
        allInfoGains = []

        totalNum = float(self.examples.shape[0])
        minForAttr = float( np.min(self.dataSet[self.examples, attrIndex]) )
        maxForAttr = float( np.max(self.dataSet[self.examples, attrIndex]) )
        allThetas = (np.random.uniform(minForAttr, maxForAttr, 5)).tolist()
    
        for theta in allThetas:
            uncertainExamples = []
            
            examplesLeft = [index for index in self.examples if self.dataSet[index][attrIndex] != -20 and 
                            self.dataSet[index][attrIndex] <= theta]
            examplesRight = [index for index in self.examples if self.dataSet[index][attrIndex] != -20 and
                             index not in examplesLeft]
            uncertainExamples = [index for index in self.examples if self.dataSet[index][attrIndex] == -20]
            
            if (len(examplesLeft) > len(examplesRight)):
                examplesLeft += uncertainExamples
            else:
                examplesRight += uncertainExamples
            
            examplesLeft = np.asarray(examplesLeft)
            examplesRight = np.asarray(examplesRight)
        
            numLeft = float(examplesLeft.shape[0])
            numRight = float(examplesRight.shape[0])
        
            if not examplesLeft.tolist():
                numClassesLeft = [0,0]
            else:
                numClassesLeft = np.bincount(np.asarray(self.dataSet[examplesLeft, 4], dtype="int"))
            if not examplesRight.tolist():
                numClassesRight = [0,0]
            else:
                numClassesRight = np.bincount(np.asarray(self.dataSet[examplesRight, 4], dtype="int"))
            
            pLeft = self._getSafeValue(numClassesLeft, 1)
            nLeft = self._getSafeValue(numClassesLeft, 0)
            pRight = self._getSafeValue(numClassesRight, 1)
            nRight = self._getSafeValue(numClassesRight, 0)
        
            entropyLeft = self.getEntropy (pLeft, nLeft)
            entropyRight = self.getEntropy (pRight, nRight)
            expectedEntropy = (((numLeft/totalNum)*entropyLeft) + ((numRight/totalNum)*entropyRight))
            allInfoGains.append(self.entropy - expectedEntropy)
    
        return allThetas[allInfoGains.index(max(allInfoGains))]

    def getEntropy(self, p, n):
        if (p + n == 0.0):
            return 0
        e1 = 0.0
        e2 = 0.0
        if (p != 0.0):
            e1 = (p / (p + n)) * math.log((p / (p + n)),2)
        if (n != 0.0):
            e2 = (n / (p + n)) * math.log((n / (p + n)),2)
    
        entropy = - (e1 + e2)  
        return entropy
    
    def printStats(self):
        print "Allowed Attributes: "+ str(self.allowedAttrs)
        print "Leaf Class: "+ str(self.leafClass)


def makeDecisionTree(dataSet, node, allowedAttrs, parentNode, level):
    if not node.examples.tolist():
        pluralityValue = np.bincount(np.asarray(dataSet[parentNode.nodeExamplesIndices, 4] ,dtype="int")).tolist()
        outputClass = pluralityValue.index(max(pluralityValue))
        return outputClass
    
    elif (np.all(np.asarray(dataSet[node.examples, 4] ,dtype="int") == np.asarray(dataSet[node.examples, 4] ,dtype="int")[0] )):
        outputClass = dataSet[node.examples, 4][0]
        return outputClass
    
    elif (len(allowedAttrs)<=1 ):
        pluralityValue = np.bincount(np.asarray(dataSet[node.examples, 4] ,dtype="int")).tolist()
        outputClass = pluralityValue.index(max(pluralityValue))
        return outputClass
    
    else:
        allowedAttrs.remove(node.bestAttr)
        examplesLeft = [index for index in node.examples if dataSet[index][node.bestAttr] != -20 and 
                                    dataSet[index][node.bestAttr] <= node.bestTheta]
        examplesRight = [index for index in node.examples if dataSet[index][node.bestAttr] != -20 and
                                     index not in examplesLeft]
        uncertainExamples = [index for index in node.examples if dataSet[index][node.bestAttr] == -20]
        
        if (len(examplesLeft) > len(examplesRight)):
            examplesLeft = examplesLeft + uncertainExamples
        else:
            examplesRight = examplesRight + uncertainExamples

        examplesLeft = np.asarray(examplesLeft)
        examplesRight = np.asarray(examplesRight)
        
        node.leftChild = Node(dataSet, examplesLeft, allowedAttrs, None)
        node.rightChild = Node(dataSet, examplesRight, allowedAttrs, None)

        node.leftChild.leafClass = makeDecisionTree(dataSet, node.leftChild, allowedAttrs[:], node, level + 1)
        node.rightChild.leafClass = makeDecisionTree(dataSet, node.rightChild, allowedAttrs[:], node, level + 1) 
    
    return []


def buildRandomForest(dataSet, attrIndices, printStats=False):
    #splitIndices = np.asarray(np.array_split(attrIndices, 10))
    splitIndices = np.asarray([np.random.randint(0, len(attrIndices), 900) for i in range(5)])
    InitAttrIndices = [i for i in range(dataSet.shape[1] - 1)]

    randomForest = []
    for splitPart in splitIndices:
        randomForest.append(DecisionTree(dataSet, splitPart, random.sample(InitAttrIndices, 2)))

    for decisionTree in randomForest:
        depth = 1
        if printStats is True:
            decisionTree.root.printStats()
        makeDecisionTree(dataSet, decisionTree.root, decisionTree.root.allowedAttrs, None, depth)

    return randomForest


def _classifyQueryWithTree (example, node):
    if (node.leafClass is not None):
        return node.leafClass
    if (example[0][node.bestAttr] == -20):
        leftChildCount = node.leftChild.examples.shape[0]
        rightChildCount = node.rightChild.examples.shape[0]
        
        if (leftChildCount > rightChildCount):
            return _classifyQueryWithTree (example, node.leftChild)
        else:
            return _classifyQueryWithTree (example, node.rightChild)
    else:
        if (example[0][node.bestAttr] <= 0.0):
            return _classifyQueryWithTree (example, node.leftChild)
        elif (example[0][node.bestAttr] > 0.0):
            return _classifyQueryWithTree (example, node.rightChild)

def classifyQuery(randomForest, data):
    result = []
    for decisionTree in randomForest:
        result.append(int(_classifyQueryWithTree ([data], decisionTree.root) ) )
    return np.argmax(np.bincount(result))

if __name__ == "__main__":
    with open("data_banknote_authentication.txt", "r") as dataset:
        bankNoteData = removeAttrsRandom(np.asarray([line.split(",") for line in dataset.readlines()], dtype=np.float64))
        
    attrIndices = [i for i in range(bankNoteData.shape[0])]
    randomForest = buildRandomForest(bankNoteData, attrIndices)
    
    print classifyQuery(randomForest, [-3.2692,-12.7406,15.5573,-0.14182])
 