# coding:utf-8
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

detailstr = [
    [u"青绿", u"乌黑", u"浅白"],
    [u"蜷缩", u"稍蜷", u"硬挺"],
    [u"浊响", u"沉闷", u"清脆"],
    [u"清晰", u"稍糊", u"模糊"],
    [u"凹陷", u"稍陷", u"平坦"],
    [u"硬滑", u"软粘"],
    [],
    []
]

feaure = [u"色泽", u"根蒂", u"敲声", u"纹理", u"脐部", u"触感", u"密度", u"含糖率"]

constr = {}
continuous_indexs = [6, 7]
colActive = [True, True, True, True, True, True, True, True]


def createDataSet():
    """
    creat dataset and label,
    """
    dataset = [
        [0, 1, 1, 0, 2, 0, 1, 1, 1, 0, 2, 2, 0, 2, 1, 2, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 1, 2, 2, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 2, 1, 1, 0, 2, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0, 0, 1, 2, 1],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
        [0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437,
         0.666, 0.243, 0.245, 0.343, 0.639, 0.657, 0.360, 0.593, 0.719],
        [0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211,
         0.091, 0.267, 0.057, 0.099, 0.161, 0.198, 0.370, 0.042, 0.103]
    ]
    labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    return dataset, labels


def getNumEntries(active):
    numEntries = 0
    for i in range(len(active)):
        if active[i]:
            numEntries += 1
    return numEntries


def calcShannonEnt(labels, active):
    """
    calcuate shannonEnt by labels
    labels 标签
    active　表明每个example是否有效
    """
    labelCount = {}
    for i, label in enumerate(labels):
        if not active[i]:
            continue
        if label not in labelCount.keys():
            labelCount[label] = 0
        labelCount[label] += 1

    numEntries = getNumEntries(active)
    shannonEnt = 0.0
    for key in labelCount.keys():
        prob = float(labelCount[key]) / float(numEntries)
        shannonEnt -= prob * math.log(prob, 2)

    return shannonEnt


def splitDataSet(dataset, active, colActive, axis, value):
    """
    dataset    数据集
    labels     标签集
    active     表明example是否有效
    axis       划分的特征号
    value　　　 特征值
    """
    if not colActive[axis]:
        sys.exit(-1)

    retActive = active[:]
    for i, featVec in enumerate(dataset):
        if not active[i]:
            continue
        if featVec[axis] != value:
            retActive[i] = False
    return retActive


def splitContinuous(dataset, active, rowActive, axis, value):
    if not colActive[axis]:
        sys.exit(-1)

    retActive1 = active[:]; retActive2 = active[:]
    for i, featVec in enumerate(dataset):
        if not active[i]:
            continue
        if featVec[axis] <= value:
            retActive2[i] = False
        else:
            retActive1[i] = False
    return [retActive1, retActive2]


count = 0
def chooseBestFeatureToSplib(dataset, labels, active, rowActive):
    global continuous_indexs
    baseEntropy = calcShannonEnt(labels, active)
    bestInfoGain = 0.0;
    bestFeaure = -1;
    bestActiveList = [];
    bestFeaureUniqueVals = set()

    global count
    print "第%d次分支计算出的Gaininfo为\n" %(count)
    count += 1
    numEntries = getNumEntries(active)
    for i in range(len(dataset[0])):
        if not rowActive[i]:
            continue
        featList = [dataset[j][i] for j in range(len(dataset)) if active[j]]
        newEntropy = 0.0

        # 需要获得的数据
        uniqueVals = set(featList)
        activeList = [];
        infoGain = 0.0
        if i not in continuous_indexs:
            for value in uniqueVals:
                tempactive = splitDataSet(dataset, active, rowActive, i, value)
                activeList.append(tempactive)
                prob = getNumEntries(tempactive) / float(numEntries)
                newEntropy += prob * calcShannonEnt(labels, tempactive)
            infoGain = baseEntropy - newEntropy
        else:
            sortedUniqueVals = sorted(uniqueVals)
            valList = [(sortedUniqueVals[j] + sortedUniqueVals[j - 1]) / 2 for j in range(len(sortedUniqueVals)) if j != 0]

            bestvalue = -1
            for value in valList:
                tempinfogain = 0.0
                tempactivelist = splitContinuous(dataset, active, rowActive, i, value)
                for tempactive in tempactivelist:
                    prob = getNumEntries(tempactive) / float(numEntries)
                    tempinfogain += prob * calcShannonEnt(labels, tempactive)
                tempinfogain = baseEntropy - tempinfogain
                if len(activeList) == 0 or tempinfogain > infoGain:
                    infoGain = tempinfogain
                    bestvalue = value
                    activeList = tempactivelist
            uniqueVals = []
            uniqueVals.append(bestvalue)

        print "Gain(%s) = %s" %(feaure[i], str(infoGain))
        if bestFeaure == -1 or infoGain > bestInfoGain:
            bestFeaure = i
            bestActiveList = activeList
            bestInfoGain = infoGain
            bestFeaureUniqueVals = uniqueVals
      #  global feaure
       # print "infoGain(%s) = %s" %(feaure[i], str(infoGain))

    return bestFeaure, bestActiveList, bestFeaureUniqueVals


def majorityCnt(labels, active):
    classCount = {}
    for i in range(len(labels)):
        if not active[i]:
            continue
        if labels[i] not in classCount.keys():
            classCount[labels[i]] = 0
        classCount[labels[i]] += 1

    maxIndex = -1; maxCount = -1
    for key in classCount:
        if maxCount == -1 or classCount[key] > maxCount:
            maxCount = classCount[key]
            maxIndex = key
    return key



def isSameClass(labels, active):
    type = -1
    for i in range(len(labels)):
        if not active[i]:
            continue
        if type == -1:
            type = labels[i]
        else:
            if labels[i] != type:
                return False

    return True

def firstExampleClass(labels, active):
    for i in range(len(labels)):
        if not active[i]:
            continue
        return labels[i]

def getFeatureNum(dataset, rowActive):
    num = 0
    for i in range(len(dataset[0])):
        if not rowActive[i]:
            continue
        num += 1
    return num

def createTree(dataset, labels, active, rowActive):
    if isSameClass(labels, active):
        return firstExampleClass(labels, active)
    if getFeatureNum(dataset, rowActive) == 1:
        return majorityCnt(labels, active)


    bestFeaure, bestActiveList, bestFeaureUniqueVals = chooseBestFeatureToSplib(dataset, labels, active, rowActive)

    #print "best feaure is %s" %(feaure[bestFeaure])

    myTree = {bestFeaure : {}}
    bestFeaureUniqueVals = list(bestFeaureUniqueVals)

    for i in range(len(bestFeaureUniqueVals)):
        newRowActive = rowActive[:]
        newRowActive[bestFeaure] = False
        value = bestFeaureUniqueVals[i]
        global continuous_indexs
        if bestFeaure in continuous_indexs:
            global detailstr
            detailstr[bestFeaure].append("<=" + str(bestFeaureUniqueVals[0]))
            detailstr[bestFeaure].append(">" + str(bestFeaureUniqueVals[0]))
            myTree[bestFeaure][0] = createTree(dataset, labels, bestActiveList[0], newRowActive)
            newRowActive2 = newRowActive[:]
            myTree[bestFeaure][1] = createTree(dataset, labels, bestActiveList[1], newRowActive2)
        else:
            myTree[bestFeaure][value] = createTree(dataset, labels, bestActiveList[i], newRowActive)
    return myTree


#def createPlot():
#    fig = plt.figure(1,facecolor='white')
#    fig.clf()



if __name__ == "__main__":
    dataset, labels = createDataSet()
    dataset = np.transpose(np.array(dataset)).tolist()
    active = [True for i in range(17)]
    rowActive = [True for i in range(8)]
    createTree(dataset, labels, active, rowActive)