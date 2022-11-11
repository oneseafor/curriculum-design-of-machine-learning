import numpy as np
import pandas as pd
import random
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import pylab as pl
import itertools
import matplotlib.pyplot as plt

#import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import os
import joblib
pl.mpl.rcParams['font.sans-serif'] = ['SimHei'] # 没有这句话汉字都是⼝⼝
dataset=pd.read_csv(r'C:\Users\86188\Desktop\data\Train.csv')
mirna_seqdf=pd.read_csv(r'C:\Users\86188\Desktop\data\mirna_seq.csv')#(['mirna', 'seq']
gene_seqdf=pd.read_csv(r'C:\Users\86188\Desktop\data\gene_seq.csv')#'label', 'sequence'
dataset_mirna=dataset['miRNA']#train获取表格中为miRNA的值
#print("dataset_mirna\n",dataset_mirna[0:10])
dataset_gene=dataset['gene']#train获取表格中为gene的值
#print("dataset_gene\n",dataset_gene[0:10])
dataset_label=dataset['label']#train获取表格中为label的值
#print("dataset_label\n",dataset_label[0:10])
gene_index=gene_seqdf['label'].values.tolist()#获取列表中的值，除去编号
#print("gene_index\n",gene_index[0:10])
gene_seq=gene_seqdf['sequence']#获取gene里的基因型)
#print("gene_seq\n",gene_seq[0:10])
mirna_index=mirna_seqdf['mirna'].values.tolist()#获取mirna中的名称
#print("mirna_index\n",mirna_index[0:10])
mirna_seq=mirna_seqdf['seq']#获取mirna中的基因型
#print("mirna_seq\n",mirna_seq[0:10])
key_set={}
key_set_T={}
#print(list(itertools.product('UCGA', repeat =3)))
for i in itertools.product('UCGA', repeat =5):#itertools.product('BCDEF', repeat = 2):#itertools.product获取数据的笛卡尔积，
    #print(i)
    obj=''.join(i)
    #print(obj)
    ky={'{}'.format(obj):0}#输出{format（obj）：0}
    #print(ky)
    key_set.update(ky)#将{format（obj）：0}键-值对添加到字典d中
    #print(key_set)
for i in itertools.product('TCGA', repeat =5):#itertools.product('BCDEF', repeat = 2):
    #print(i)
    obj=''.join(i)
   # print(obj)
    ky={'{}'.format(obj):0}
    key_set_T.update(ky)
    
def clean_key_set(key_set):#清除数据，将key值变为0
    for i,key in enumerate(key_set):#enumerate多用于在for循环中得到计数，利用它可以同时获得索引和值
    #print(i,key,key_set[key])
        key_set[key]=0
    return key_set
def return_features(n,seq):#返回特征值
    clean_key_set(key_set)#清除key值，将key值变成0
    #print("key_set",key_set)
    key=key_set
    
    if '\n' in seq:#数据存在噪声，除去\n
        #print("len(seq)",seq)
        seq=seq[0:-1]
    for i in range(n,len(seq)+1-n):
        win=seq[i:i+n]
        #print("i",i)
        #print("win",win)
        ori=key_set['{}'.format(win)]#判断是否在片段基因是否在key_set中，即获取key_set中获取key值
        #print("ori",ori)
        key_set['{}'.format(win)]=ori+1#记录遍历过的基因片段
        #print("key_set",key_set)
    #print("key_set",key_set)
    return key_set
def return_gene_features(n,seq):#返回特征值
    clean_key_set(key_set_T)
    key=key_set_T
    if '\n' in seq:
        seq=seq[0:-1]
    for i in range(n,len(seq)+1-n):
        win=seq[i:i+n]
        #print(win)
        ori=key_set_T['{}'.format(win)]
        key_set_T['{}'.format(win)]=ori+1
    return key_set_T
#print(len(dataset_mirna))
def construct_dataset(dataset_mirna,dataset_gene):#使用拼接方法构建数据集
    list_mirna_feature=[]
    list_gene_feature=[]
    for i in range(0,len(dataset_mirna)):
        try:
            mirna=dataset_mirna[i]#获取测试集第一个数据
            m_index=mirna_index.index(mirna)#在mirna中找到该数据对应的索引
            #print("m_index",m_index)
            mirna_f=return_features(5,mirna_seq[m_index]) #找到该数据对应的基因型。并对特征值进行计算
            gene=dataset_gene[i]#获取测试的第一个gene数据
            g_index=gene_index.index(gene)#获取第一个数据在geneindex中的索引
            gene_f=return_gene_features(5, gene_seq[g_index])#获取该数据对应的基因型，并对特征值进行计算
            mirna_feature=mirna_f.copy()#复制一个数组
            #print("mirna_feature",mirna_feature)
            gene_feature=gene_f.copy()
            list_mirna_feature.append(mirna_feature)#将mirna的特征数组加入到list_mirna_feature数组中
            #print("list_mirna_feature",list_mirna_feature)
            list_gene_feature.append(gene_feature)
            #print(list_gene_feature)
        except:
            mirna=dataset_mirna[i]
            gene=dataset_gene[i]
            print('error detected',i,mirna,gene)
    lmpd=pd.DataFrame(list_mirna_feature)#转化为表格，获取出现次数之和
    #print("lmpd",lmpd)
    lgpd=pd.DataFrame(list_gene_feature)
    #print("lgpd",lgpd)
    X=pd.concat([lmpd,lgpd],axis=1)#将表格合并,数据进行拼接
    #print("X",X)
    return X
    #return X
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
# X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# Y = pd.DataFrame(data=iris.target)
#data = pd.concat([X, Y], axis=1);
data=X
b=X
#print(data)
print("X",X)

# 训练随机森林
M = []  # 存储决策树模型的数组
R = []  # 存储各决策树随机选取的特征组合
n_trees = 3  # 设置森林中树的颗数
# 训练多棵树
for i in range(n_trees):
    # 随机选取样本
    sample = data.sample(frac=0.7)  # 对样本进行采样，目的是建造不同的树用于  sample()从DataFrame中随机选择行和列
    # 随机选取特征,随机选取k个特征组合r
    k = np.random.randint(1, sample.shape[1])  # 随机选取k个特征
    r = np.random.choice(range(sample.shape[1]), k, replace=False).tolist()  # replace=False 无放回的随机选取2个特征组合
    X = sample.iloc[:, r]
    print("simple",sample)
    print("X",X)

    # 选取Y
    Y = sample.iloc[:, -1]
    print("Y",Y)

    # 新建决策树模型
    model = DecisionTreeClassifier()
    model.fit(X, Y)

    # 存储模型
    M.append(model)  # 将决策树模型加入数组
    R.append(r)  # 将特征组合加入数组中
    #print('第' + str(i) + '颗预测score=', model.score(X, Y))  # 打印每个基础模型的效果
# 测试随机森林，将每个模型预测值的结果添加到result(DataFrame)中
#print("M",M)
#print("R",R)
p=[]
for i in range(185):
    p.append(0)
    
df_predict=pd.read_csv(r'C:\Users\86188\Desktop\data\test_dataset.csv')
predict_mirna=df_predict['miRNA']
predict_gene=df_predict['gene']
#测试数据生成器
X_predict=construct_dataset(predict_mirna,predict_gene)
X_predict['匹配']=p
result = pd.concat([pd.DataFrame([M[i].predict(data.iloc[:, R[i]])]) for i in range(n_trees)], ignore_index=True)
# 输出预测结果,取众数作为最终对每个样本标签的预测值
predict = result.mode(axis=0).values[0].astype(int)
print('预测值结果=', predict)
print("predict",predict)
# 计算准确率
#score = sum(np.where(predict == a, 1, 0)) / len(data)
#print(score)

