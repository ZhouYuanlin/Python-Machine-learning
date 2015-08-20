# -*-  utf-8 -*-
#计算给定数据集的香农熵
from math import log

def calcShannonEnt(dataSet):
	numEntries = len(dataSet)
	laberCounts = {}
	#为所有可能分类创建字典
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in laberCounts.key:
			laberCounts[currentLabel] = 0
		laberCounts[currentLabel] += 1
	shannonEnt = 0.0
	for key in laberCounts:
		prob = float(labelCounts[key])/numEntries
		#以2为底求对数
		shannonEnt -= prob * log(prob, 2)
	return shannonEnt