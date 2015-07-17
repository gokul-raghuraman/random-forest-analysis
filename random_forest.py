import numpy as np
import math
import random

class DecisionTree:
    def __init__(self, dataSet, examples, attributes):
        self.root = Node(dataSet, examples, attributes, -1)


class Node:
    leftChild = None
    rightChild = None
    
    def __init__(self, dataSet, examples, allowedAttrs, leafClass):
        self.dataSet = dataSet
        self.examples = np.asarray(examples)
        self.allowedAttrs = allowedAttrs
        self.leafClass = leafClass 
        
        if (self.leafClass == -1):
            n = self.getClassCount(0)
            p = self.getClassCount(1)
            self.entropy = self.getEntropy(p, n)
            (self.bestAttr, self.bestTheta) = self.getBestAttr()

    def getClassCount(self, cls):
        countArray = np.bincount(np.asarray(self.dataSet[self.examples, 4], dtype="int"))
        try:
            num = float(countArray[cls])
        except IndexError: 
            num = 0.0
        return num

    def getBestAttr (self):
        totalNum = float(self.examples.shape[0])
        allInfoGains = []
        for attrIndex in self.allowedAttrs:
            bestTheta = self.getBestTheta (attrIndex)
            
            examplesLeft = np.asarray([index for index in self.examples if self.dataSet[index][attrIndex] <= bestTheta])
            examplesRight = np.asarray([index for index in self.examples if index not in examplesLeft])
            
            numLeft = float(examplesLeft.shape[0])
            numRight = float(examplesRight.shape[0])
            if not examplesLeft.tolist():
                numClassesLeft = [0, 0]
            else:
                numClassesLeft = np.bincount(np.asarray(self.dataSet[examplesLeft, 4], dtype="int"))
            if not examplesRight.tolist():
                numClassesRight = [0, 0]
            else:
                numClassesRight = np.bincount(np.asarray(self.dataSet[examplesRight, 4], dtype="int"))
        
            pLeft = self._getSafeValue(numClassesLeft, 1)
            nLeft = self._getSafeValue(numClassesLeft, 0)
            pRight = self._getSafeValue(numClassesRight, 1)
            nRight = self._getSafeValue(numClassesRight, 0)
        
            entropyLeft = self.getEntropy(pLeft, nLeft)
            entropyRight = self.getEntropy(pRight, nRight)
            expectedEntropy = (numLeft/totalNum) * entropyLeft + (numRight/totalNum) * entropyRight
            allInfoGains.append([self.entropy - expectedEntropy, bestTheta])
            
        bestAttr = self.allowedAttrs[allInfoGains.index(max(allInfoGains))]
        bestTheta = allInfoGains[allInfoGains.index(max(allInfoGains))][1]
        return [bestAttr,  bestTheta]
      
    def getBestTheta (self, attrIndex):
        totalNum = float(self.examples.shape[0])
        minForAttr = float(np.min(self.dataSet[self.examples, attrIndex]))
        maxForAttr = float(np.max(self.dataSet[self.examples, attrIndex]))
        allThetas = (np.random.uniform(minForAttr, maxForAttr, 5)).tolist()
        allInfoGains = []
        for theta in allThetas:
                        
            examplesLeft = np.asarray([index for index in self.examples if self.dataSet[index][attrIndex] <= theta])
            examplesRight = np.asarray([index for index in self.examples if index not in examplesLeft])
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
        
            entropyLeft = self.getEntropy(pLeft, nLeft)
            entropyRight = self.getEntropy(pRight, nRight)
            expectedEntropy = (numLeft / totalNum) * entropyLeft + (numRight / totalNum) * entropyRight
            allInfoGains.append(self.entropy  - expectedEntropy)
        
        return allThetas[allInfoGains.index(max(allInfoGains))]
    
    def getEntropy(self, p, n):
        if (p + n == 0.0):
            return 0
        e1 = 0.0
        e2 = 0.0
        if (p != 0.0):
            e1 = (p / (p + n)) * math.log((p / (p + n)), 2)
        if (n != 0.0):
            e2 = (n / (p + n)) * math.log((n / (p + n)), 2)
    
        entropy = - (e1 + e2)  
        return entropy
     
    @staticmethod
    def _getSafeValue(array, index):
        try:
            val = float(array[index])
        except IndexError:
            val = 0.0
        return val    

    def printStats(self):
        print("Examples           : %s" % self.examples)
        print("Allowed Attributes : %s" % self.allowedAttrs)
        print("Leaf Class         : %s" % self.leafClass)

def makeDecisionTree(dataSet, node, allowedAttrs, parent, level):
    if not node.examples.tolist():
        pluralityValue = np.bincount(np.asarray(dataSet[parent.examples, 4] ,dtype="int")).tolist()
        outputClass = pluralityValue.index(max(pluralityValue))
        return outputClass
    
    elif (np.all(np.asarray(dataSet[node.examples, 4], dtype="int") == np.asarray(dataSet[node.examples, 4] ,dtype="int")[0])):
        outputClass = dataSet[node.examples, 4][0]
        return outputClass
    
    elif (len(allowedAttrs) <= 1):
        pluralityValue = np.bincount(np.asarray(dataSet[node.examples, 4], dtype="int")).tolist()
        outputClass = pluralityValue.index(max(pluralityValue))
        return outputClass

    else:
        allowedAttrs.remove(node.bestAttr)
        examplesLeft = np.asarray([index for index in node.examples if dataSet[index][node.bestAttr] <= node.bestTheta])
        examplesRight = np.asarray([index for index in node.examples if index not in examplesLeft])
        node.leftChild = Node(dataSet, examplesLeft, allowedAttrs, -1)
        node.rightChild = Node(dataSet, examplesRight, allowedAttrs, -1)
        node.leftChild.leafClass = makeDecisionTree(dataSet, node.leftChild, allowedAttrs[:], node, level + 1) 
        node.rightChild.leafClass = makeDecisionTree(dataSet, node.rightChild, allowedAttrs[:], node, level + 1)
    
    return []

def buildRandomForest(dataSet, attrIndices, printStats=False):
    np.random.shuffle(attrIndices) #shuffle indices of rows so trees are not biased
    #splitAttrIndices = np.asarray(np.array_split(attrIndices, 3))
    splitAttrIndices = np.asarray([np.random.randint(0, len(attrIndices), 900) for i in range(5)])
    InitAttrIndices = [i for i in range(dataSet.shape[1] - 1)]
    randomForest = []
    for splitPart in splitAttrIndices:
        randomForest.append(DecisionTree(dataSet, splitPart, random.sample(InitAttrIndices, 2)))
            
    for decisionTree in randomForest:
        depth = 1
        if printStats is True:
            decisionTree.root.printStats()
        makeDecisionTree(dataSet, decisionTree.root, decisionTree.root.allowedAttrs, None, depth)
    return randomForest


def _classifyQueryWithTree (example, node):
    if (node.leafClass != -1):
        return node.leafClass
    else:
        if (example[0][node.bestAttr]<=node.bestTheta):
            return _classifyQueryWithTree (example, node.leftChild)
        elif (example[0][node.bestAttr]>node.bestTheta):
            return _classifyQueryWithTree (example, node.rightChild)


def classifyQuery(randomForest, data):
    results = []
    for decisionTree in randomForest:
        results.append(int(_classifyQueryWithTree ([data], decisionTree.root) ) )
    
    return np.argmax(np.bincount(results)) 

if __name__ == "__main__":
    with open("data_banknote_authentication.txt", "r") as dataset:
        bankNoteData = np.asarray([line.split(",") for line in dataset.readlines()], dtype=np.float64)
    attrIndices = [i for i in range(bankNoteData.shape[0])]
    randomForest = buildRandomForest(bankNoteData, attrIndices)
    print classifyQuery(randomForest, [1.645,7.8612,-0.87598,-3.5569])
