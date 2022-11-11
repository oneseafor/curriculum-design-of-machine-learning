# -*- coding: utf-8 -*-
import csv
from random import seed
from random import randrange
#from math import sqrt
import pandas as pd
#import numpy as np
import pylab as pl
import itertools

pl.mpl.rcParams['font.sans-serif'] = ['SimHei'] # 没有这句话汉字都是⼝⼝
####构建数据集
dataset=pd.read_csv(r'C:\Users\86188\Desktop\data\Train.csv')
mirna_seqdf=pd.read_csv(r'C:\Users\86188\Desktop\data\mirna_seq.csv')#(['mirna', 'seq']
gene_seqdf=pd.read_csv(r'C:\Users\86188\Desktop\data\gene_seq.csv')#'label', 'sequence'
dataset_mirna=dataset['miRNA']#train获取表格中为miRNA的值
dataset_gene=dataset['gene']#train获取表格中为gene的值
dataset_label=dataset['label']#train获取表格中为label的值
gene_index=gene_seqdf['label'].values.tolist()#获取列表中的值，除去编号
gene_seq=gene_seqdf['sequence']#获取gene里的基因型)
mirna_index=mirna_seqdf['mirna'].values.tolist()#获取mirna中的名称
mirna_seq=mirna_seqdf['seq']#获取mirna中的基因型
key_set={}
key_set_T={}
for i in itertools.product('UCGA', repeat =3):#itertools.product('BCDEF', repeat = 2):#itertools.product获取数据的笛卡尔积，

    obj=''.join(i)
    
    ky={'{}'.format(obj):0}#输出{format（obj）：0}
    
    key_set.update(ky)#将{format（obj）：0}键-值对添加到字典d中
    
for i in itertools.product('TCGA', repeat =3):#itertools.product('BCDEF', repeat = 2):
   
    obj=''.join(i)
   
    ky={'{}'.format(obj):0}
    key_set_T.update(ky)
    
def clean_key_set(key_set):#清除数据，将key值变为0
    for i,key in enumerate(key_set):#enumerate多用于在for循环中得到计数，利用它可以同时获得索引和值
    
        key_set[key]=0
    return key_set
def return_features(n,seq):#返回特征值
    clean_key_set(key_set)#清除key值，将key值变成0
    
    key=key_set
    
    if '\n' in seq:#数据存在噪声，除去\n
        
        seq=seq[0:-1]
    for i in range(n,len(seq)+1-n):
        win=seq[i:i+n]
        
        ori=key_set['{}'.format(win)]#判断是否在片段基因是否在key_set中，即获取key_set中获取key值
        
        key_set['{}'.format(win)]=ori+1#记录遍历过的基因片段
    
    return key_set
def return_gene_features(n,seq):#返回特征值
    clean_key_set(key_set_T)
    key=key_set_T
    if '\n' in seq:
        seq=seq[0:-1]
    for i in range(n,len(seq)+1-n):
        win=seq[i:i+n]
        
        ori=key_set_T['{}'.format(win)]
        key_set_T['{}'.format(win)]=ori+1
    return key_set_T
def construct_dataset(dataset_mirna,dataset_gene):#使用拼接方法构建数据集
    list_mirna_feature=[]
    list_gene_feature=[]
    for i in range(0,len(dataset_mirna)):
        try:
            mirna=dataset_mirna[i]#获取测试集第一个数据
            m_index=mirna_index.index(mirna)#在mirna中找到该数据对应的索引
            mirna_f=return_features(3,mirna_seq[m_index]) #找到该数据对应的基因型。并对特征值进行计算
            gene=dataset_gene[i]#获取测试的第一个gene数据
            g_index=gene_index.index(gene)#获取第一个数据在geneindex中的索引
            gene_f=return_gene_features(3, gene_seq[g_index])#获取该数据对应的基因型，并对特征值进行计算
            mirna_feature=mirna_f.copy()#复制一个数组
            gene_feature=gene_f.copy()
            list_mirna_feature.append(mirna_feature)#将mirna的特征数组加入到list_mirna_feature数组中

            list_gene_feature.append(gene_feature)
        except:
            mirna=dataset_mirna[i]
            gene=dataset_gene[i]
            print('error detected',i,mirna,gene)
    lmpd=pd.DataFrame(list_mirna_feature)#转化为表格，获取出现次数之和
    lgpd=pd.DataFrame(list_gene_feature)
    X=pd.concat([lmpd,lgpd],axis=1)#将表格合并,数据进行拼接
    return X
#标签换为数字
Y=[]
for i,label in enumerate(dataset_label):
    if label =='Functional MTI':
        Y.append(1)
    else:
        Y.append(0)
X=construct_dataset(dataset_mirna,dataset_gene)
a=Y
X['匹配']=a
outputpath='C:/Users/86188/Desktop/data1/DataFrame.csv'
X.to_csv(outputpath,sep=',',index=False,header=True)



 
##随机森林算法 ##
def loadCSV(filename):#加载数据，一行行的存入列表
    dataSet = []
    with open(filename, 'r') as file:
        csvReader = csv.reader(file)
        for line in csvReader:
            dataSet.append(line)
    return dataSet[1:]
 
# 除了标签列，其他列都转换为float类型
def column_to_float(dataSet):
    featLen = len(dataSet[0]) - 1
    for data in dataSet:
        for column in range(featLen):
            data[column] = float(data[column].strip())#strip()函数，将其他的字符去掉，保证数据的干净
 
# 将数据集随机分成N块，方便交叉验证，其中一块是测试集，其他四块是训练集
#构建一个三维数组，第一维表示训练集和测试集的个数，第二维表示新数据集中导入数据的个数，第三维即表示数据。
def spiltDataSet(dataSet, n_folds):
    fold_size = int(len(dataSet) / n_folds)#获取每一个数据集的长度
    print("fold_size",fold_size)
    #dataSet_copy = list(dataSet)
    dataSet_copy = dataSet
    dataSet_spilt = []
    for i in range(n_folds):
        fold = []
        while len(fold) < fold_size:  # 这里不能用if，if只是在第一次判断时起作用，while执行循环，直到条件不成立
            index = randrange(len(dataSet_copy))
            fold.append(dataSet_copy.pop(index))  # pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。
        dataSet_spilt.append(fold)
    return dataSet_spilt
 
# 构造数据子集,使用随机分块的数据进行训练
def get_subsample(dataSet, ratio):#radio的值为1.0，
    subdataSet = []
    #print("len(dataSet)",len(dataSet))
    #print("len(dataSet) * ratio",len(dataSet) * ratio)
    lenSubdata = round(len(dataSet) * ratio)#返回浮点数,round函数：四舍五入的功能
    #print("lenSubdata",lenSubdata)
    while len(subdataSet) < lenSubdata:
        index = randrange(len(dataSet) - 1)
        subdataSet.append(dataSet[index])
    return subdataSet#f返回一个新的的数据集
 
# 分割数据集
def data_spilt(dataSet, index, value):#输入训练集   随机特征值的下标    随机下标对应每一行中的值
    left = []
    right = []
    for row in dataSet:
        if row[index] < value:
            #如果这一行的对应下标的值小于传入随机获取的下标的值，这一行放入left，否则放入right
            left.append(row)
        else:
            right.append(row)
    return left, right#返回left与right值
 
# 计算分割代价
def spilt_loss(left, right, class_values):
    loss = 0.0
    for class_value in class_values:#第一次取0
        left_size = len(left)
        if left_size != 0:  # 防止除数为零
            prop = [row[-1] for row in left].count(class_value) / float(left_size)
            #prop为获取为0的数目占总left数据的比值
            loss += (prop * (1.0 - prop))#若占比为1，即loss为0，若占比为0.5，即loss为0.25
        right_size = len(right)
        if right_size != 0:
            prop = [row[-1] for row in right].count(class_value) / float(right_size)
            loss += (prop * (1.0 - prop))
            #计算right的那边的值，取值最大的情况就是两边都相等，两边的标签都是一样的，即损失值为0，同时可以确定这是最小的损失的特征值
    return loss
 
# 选取任意的n个特征，在这n个特征中，选取分割时的最优特征
def get_best_spilt(dataSet, n_features):
    features = []
    #print("set(row[-1] for row in dataSet)",set(row[-1] for row in dataSet))
    class_values = list(set(row[-1] for row in dataSet))#set函数:创建一个无序不重复元素集，获取标签，即{1，0}
    print("row[-1] for row in dataSet",set(row[-1] for row in dataSet))
    b_index, b_value, b_loss, b_left, b_right = 999, 999, 999, None, None#数据初始化
    while len(features) < n_features:
        index = randrange(len(dataSet[0])-2)
        #print("index",index)
        #一共有129列数据，现在除去最后一个，数据一共有128个下表为0到127， 随机选取n个值作为特征值的索引值
        if index not in features:
            features.append(index)#feature中为索引值
    #print( 'index000',index)
    for index in features:#找到列的最适合做节点的索引，（损失最小）
        for row in dataSet:#选择训练集中行属性
            left, right = data_spilt(dataSet, index, row[index])#以它为节点的，左右分支
            loss = spilt_loss(left, right, class_values)
            if loss < b_loss:#寻找最小分割代价
                b_index, b_value, b_loss, b_left, b_right = index, row[index], loss, left, right
    # print b_loss
    # print type(b_index)
    return {'index': b_index, 'value': b_value, 'left': b_left, 'right': b_right}
 
# 决定输出标签
def decide_label(data):
    output = [row[-1] for row in data]#将data中的标签传出
    #print("output",output)
    #print("key",len(output))#count() 方法用于统计某个元素在列表中出现的次数
    #print("out",output.count("0"))
    print("set(output)",output.count)
    #print("label",max(set(output), key=output.count("0,1")))
    return max(set(output), key=output.count)#不明白这个count属性
 
 
# 子分割，不断地构建叶节点的过程
def sub_spilt(root, n_features, max_depth, min_size, depth):
    left = root['left']#root中left属性
    # print left
    right = root['right']#root中right属性
    del (root['left'])#del函数，删除所选择的属性列，即新的数据集中不包含left属性
    del (root['right'])#同理
    # print depth
    if not left or not right:
        #如果left或者right中没有数据，即满足损失最小的时候，选中的值在索引和对应的行的值是所有行中最小的
        root['left'] = root['right'] = decide_label(left + right)#在这个情况下left加right等于训练集
        print("decide_label(left + right)",decide_label(left + right))
        # print 'testing'
        return
    if depth > max_depth:
        root['left'] = decide_label(left)
        print("decide_label(left)",decide_label(left))
        root['right'] = decide_label(right)
        print("decide_label(right)",decide_label(right))
        return
    if len(left) < min_size:
        root['left'] = decide_label(left)
    else:
        print("root=left")
        root['left'] = get_best_spilt(left, n_features)
        # print 'testing_left'
        sub_spilt(root['left'], n_features, max_depth, min_size, depth + 1)
    if len(right) < min_size:
        root['right'] = decide_label(right)
    else:
        print("root=right")
        root['right'] = get_best_spilt(right, n_features)
        # print 'testing_right'
        sub_spilt(root['right'], n_features, max_depth, min_size, depth + 1)
 
        # 构造决策树
def build_tree(dataSet, n_features, max_depth, min_size):#输入训练集，特征值，最大深度，最小规模
    root = get_best_spilt(dataSet, n_features)#训练集，特征值数目
    #root为最小损失函数时所取的index，特征值，左右子树中的数据
    sub_spilt(root, n_features, max_depth, min_size, 1)
    return root#root{'index': b_index, 'value': b_value, 'left': b_left, 'right': b_right}
# 预测测试集结果
def predict(tree, row):
    predictions = []
    if row[tree['index']] < tree['value']:
        if isinstance(tree['left'], dict):
            return predict(tree['left'], row)
        else:
            return tree['left']
    else:
        if isinstance(tree['right'], dict):
            return predict(tree['right'], row)
        else:
            return tree['right']
            # predictions=set(predictions)
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    #print("",)
    return max(set(predictions), key=predictions.count)
# 创建随机森林
def random_forest(train, test, ratio, n_feature, max_depth, min_size, n_trees):
    trees = []
    for i in range(n_trees):
        #print("i",i)
        train = get_subsample(train, ratio)#从切割的数据集中选取子集，即随机生成一部分数据集
        tree = build_tree(train, n_features, max_depth, min_size)#构建决策树
        # print 'tree %d: '%i,tree
        trees.append(tree)   
    # predict_values = [predict(trees,row) for row in test]
    predict_values = [bagging_predict(trees, row) for row in test]
    print("predict_values",predict_values)
    return predict_values
# 计算准确率
def accuracy(predict_values, actual):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predict_values[i]:
            correct += 1
    return correct / float(len(actual))
 
 
if __name__ == '__main__':
    seed(1) 
    #dataSet = loadCSV('C:/Users/shadow/Desktop/组会/sonar-all-data.csv')
    dataSet = loadCSV('C:/Users/86188/Desktop/data1/DataFrame.csv')
    column_to_float(dataSet)#dataSet
    print(len(dataSet))
    n_folds = 5
    max_depth = 16
    min_size = 1
    ratio = 1.0
    # n_features=sqrt(len(dataSet)-1)
    n_features = 1
    n_trees = 2
    folds = spiltDataSet(dataSet, n_folds)#先是切割数据集
    #print("3",folds)
    scores = []
    for fold in folds:
        # 此处不能简单地用train_set=folds，这样用属于引用,那么当train_set的值改变的时候，folds的值也会改变，所以要用复制的形式。（L[:]）能够复制序列，D.copy() 能够复制字典，list能够生成拷贝 list(L)
        train_set = folds[:]  #获取随机分块的数据集
        train_set.remove(fold)#选好训练集，即remove掉一个我们需要测试的数据集
        train_set = sum(train_set, [])  # 将多个fold列表组合成一个train_set列表,既将三维化二维
        test_set = []
        for row in fold:
            row_copy = list(row)
            row_copy[-1] = None#将最后一个数据进行修改，定义测试集
            test_set.append(row_copy)
            # for row in test_set:
            # print row[-1]  
        actual = [row[-1] for row in fold]#获取最后一行的数据作为评判的值
        #print("actual",actual)
        predict_values = random_forest(train_set, test_set, ratio, n_features, max_depth, min_size, n_trees)
        accur = accuracy(predict_values,actual)
        #print("predict_values",predict_values)
        scores.append(accur)
        #print("so",scores)
    print ('Trees is %d' % n_trees)
    print ('scores:%s' % scores)
    print ('mean score:%s' % (sum(scores) / float(len(scores))))
