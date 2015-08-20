# -*- coding:utf-8 -*-
from numpy import *
import operator
from os import listdir

#测试参数一个数据集和一个标签
def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group, labels

#用 K-近邻算法实现分类器，inX 为待分类集，dataSet 为训练样本集
def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize,1))-dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1),reverse= True)
	return sortedClassCount[0][0]

# 处理文件数据
def file2matrix(filename):
	fin = open(filename)
	arrayOLines = fin.readlines()
	numberOfLines = len(arrayOLines)
	returnMat = zeros((numberOfLines, 3))
	index = 0
	classLabelVector = []
	for line in arrayOLines:
		listFormline = line.strip().split('\t')
		returnMat[index, : ] = listFormline[:3]
		classLabelVector.append(int(listFormline[-1]))
		index += 1
	return returnMat, classLabelVector

#处理数据
def autoNorm(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDateSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDateSet = dataSet - tile(minVals, (m,1))
	normDateSet = normDateSet / (tile(ranges, (m,1)))
	return normDateSet, ranges, minVals

#测试分类器
def datingClassTest():
	hoRatio = 0.30
	datingDataMat, datingLabels = file2matrix('/Users/mumutongxue/Downloads/python/Python Machine learning/machinelearninginaction/Ch02/datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
		print "the classifire came back with: %d, the real answer is: %d"%(classifierResult, datingLabels[i])
		if(classifierResult != datingLabels[i]): errorCount += 1.0
	print "the total error rate is : %f"%(errorCount/float(numTestVecs))

# 应用
def classifyPerson():
	resultList = ['not at all', 'in small doses', 'in large doses']
	percentTats = float(raw_input("percentage of time spent playing video games?"))
	ffMiles = float(raw_input("frequent flier miles earned per year?"))
	iceCream = float(raw_input("liters of ice cream consumed per year?"))
	datingDataMat, datingLabels = file2matrix('/Users/mumutongxue/Downloads/python/Python Machine learning/machinelearninginaction/Ch02/datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = array([ffMiles, percentTats, iceCream])
	inArr = (inArr - minVals)/ranges
	classifierResult = classify0(inArr, normMat,datingLabels,3)
	print "You will probably like this person:",resultList[classifierResult-1]

# 手写识别系统===================

# 准备数据：将图像转换为测试向量

def img2vector(filename):
	returnVect = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0,32*i+j] = int(lineStr[j])
		return returnVect

#手写数字识别系统的测试代码
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))





				
