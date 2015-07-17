import numpy as np
import math


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
            n = self.getClassCount(0)
            p = self.getClassCount(1)
            self.entropy = self.getEntropy (p, n)
            self.bestAttr = self.getBestAttr (self.examples, self.allowedAttrs)
    
    def getClassCount(self, cls):
        countArray = np.bincount(np.asarray(self.dataSet[self.examples, 4], dtype="int"))
        try:
            num = float(countArray[cls])
        except IndexError: 
            num = 0.0
        return num

    def getBestAttr (self, exampleIndices, allowedAttrs):
        allInfoGains = []
    
        totalNum = float(exampleIndices.shape[0])
    
        for attrIndex in allowedAttrs:
            examplesLeft = np.asarray([index for index in exampleIndices if self.dataSet[index][attrIndex] <= 0.0])
            examplesRight = np.asarray([index for index in exampleIndices if index not in examplesLeft])
        
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
            nRight = self._getSafeValue(numClassesLeft, 0)

            entropyLeft = self.getEntropy(pLeft, nLeft)
            entropyRight = self.getEntropy(pRight, nRight)
            expectedEntropy = (numLeft / totalNum) * entropyLeft + (numRight / totalNum) * entropyRight
            allInfoGains.append(self.entropy - expectedEntropy)
        
        return allowedAttrs[allInfoGains.index(max(allInfoGains))]

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

    @staticmethod
    def _getSafeValue(array, index):
        try:
            val = float(array[index])
        except IndexError:
            val = 0.0
        return val    
    
    def printStats(self, index, depth):
        print("\nTree Depth  : %s" % depth)
        if (index == 0):
            print("left Child: ")
        elif (index == 1):
            print("right Child: ")
        print("\tAttributes allowed: %s" % ", ".join([str(attr) for attr in self.allowedAttrs]))
        print("\tLeaf Class        : %s" % self.leafClass)
        print("\tnodeEntropy       : %s" % self.entropy)
        print("\tbestAttr          : %s" % self.bestAttr)
    
    
    
def makeDecisionTree(dataSet, node, allowedAttrs, parent, depth=0):
    if not node.examples.tolist():
        pluralityValue = np.bincount(np.asarray(dataSet[parent.examples, 4] ,dtype="int")).tolist()
        outputClass = pluralityValue.index(max(pluralityValue))
        return outputClass
    
    elif len(allowedAttrs) <= 1:
        pluralityValue = np.bincount(np.asarray(dataSet[node.examples, 4] ,dtype="int")).tolist()
        outputClass = pluralityValue.index(max(pluralityValue))
        return outputClass
    
    elif np.all(np.asarray(dataSet[node.examples, 4] ,dtype="int") 
                 == np.asarray(dataSet[node.examples, 4] ,dtype="int")[0]):
        outputClass = dataSet[node.examples, 4][0] 
        return outputClass
    
    else:
        allowedAttrs.remove(node.bestAttr)

        examplesLeft = np.asarray([index for index in node.examples if dataSet[index][node.bestAttr] <= 0.0])
        examplesRight = np.asarray([index for index in node.examples if index not in examplesLeft])
        leftNode = Node(dataSet, examplesLeft, allowedAttrs, None)
        rightNode = Node(dataSet, examplesRight, allowedAttrs, None)
        node.leftChild=leftNode
        node.rightChild=rightNode
        
        leftNode.leafClass = makeDecisionTree(dataSet, node.leftChild, allowedAttrs[:], node, depth+1) 
        rightNode.leafClass = makeDecisionTree(dataSet, node.rightChild, allowedAttrs[:], node, depth+1)


def printTree(node, index, depth=0):
    node.printStats(index, depth)
    
    if(node.leftChild is not None):
        printTree(node.leftChild, 0, depth + 1)
    if(node.rightChild is not None):
        printTree(node.rightChild, 1, depth + 1)


def classifyQuery (query, node):
    if (node.leafClass is not None):
        return node.leafClass
    else:
        if (query[node.bestAttr] <= 0.0):
            return classifyQuery (query, node.leftChild)
        elif (query[node.bestAttr] > 0.0):
            return classifyQuery (query, node.rightChild)



if __name__ == '__main__':
    
    with open("data_banknote_authentication.txt", "r") as dataset:
        bankNoteData = np.asarray([line.split(",") for line in dataset.readlines()], dtype=np.float64)
    
    attrIndices = [i for i in range(bankNoteData.shape[1] - 1)]
    exampleIndices = [i for i in range(bankNoteData.shape[0])]

    #Initialize Tree with all data
    decisionTree = DecisionTree(bankNoteData, exampleIndices, attrIndices)
    
    makeDecisionTree(bankNoteData, decisionTree.root, decisionTree.root.allowedAttrs, None)

    printTree(decisionTree.root, 0)   


    query = [-2.2501,3.3129,-0.88369,-2.8974]

    print(classifyQuery (query, decisionTree.root))
