import numpy as np

with open("./data/example3.1.csv") as f:
    lines = f.readlines()
    data = np.array([[int(j) for j in i.replace("\n","").split(",")] for i in lines])



class Node:
    """
    定义节点类：
    val：节点中的实例点
    dim：当前节点的分割维度
    left：节点的左子树
    right：节点的右子树
    parent：节点的父节点
    """

    def __init__(self, val=None, dim=None, left=None, right=None, parent=None):
        self.val = val
        self.dim = dim
        self.left = left
        self.right = right
        self.parent = parent


class kdTree:
    """
    定义树类：
    dataNum：训练集的样本数量
    root：构造的kd树的根节点
    """

    def __init__(self, dataSet):
        self.dataNum = 0
        self.root = self.buildKdTree(dataSet)  ## 注意父节点的传值。

    def buildKdTree(self, dataSet, parentNode=None):
        dataNum, dimNum = data.shape  # 训练集的样本数，单个数据的维数
        if dataNum == 0:  # 如果训练集为数据，返回None
            return None
        varList = self.getVar(data)  # 计算各维度的方差
        mid = dataNum // 2  # 找到中位数的索引
        maxVarDimIndex = varList.index(max(varList))  # 找到方差最大的维度
        sortedDataIndex = data[:, maxVarDimIndex].argsort()  # 按照方差最大的维度进行排序
        midDataIndex = sortedDataIndex[mid]  # 找到该维度处于中间位置的数据，作为根节点
        if dataNum == 1:  # 如果只有一个数据，那么直接返回根节点就行
            self.dataNum = dataNum
            return Node(val=data[midDataIndex], dim=maxVarDimIndex, left=None, right=None, parent=parentNode)
        root = Node(data[midDataIndex], maxVarDimIndex, parent=parentNode, )
        """
        划分左子树和右子树，然后递归
        """
        leftDataSet = data[sortedDataIndex[:mid]]  # 注意是mid而不是不是midDataIndex
        rightDataSet = data[sortedDataIndex[mid + 1:]]
        root.left = self.buildKdTree(leftDataSet, parentNode=root)
        root.right = self.buildKdTree(rightDataSet, parentNode=root)
        self.dataNum = dataNum  # 记录训练记得样本数
        return root

    def getVar(self, data):  # 求方差函数
        rowLen, colLen = data.shape
        varList = []
        for i in range(colLen):
            varList.append(np.var(data[:, i]))
        return varList

Tree = kdTree(data)