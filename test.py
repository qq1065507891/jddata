import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn import metrics
import joblib
from bayes_opt import BayesianOptimization
from gensim import models


max_length = 500  # 表示样本表示最大的长度,表示降维之后的维度
sentence_max_length = 1500  # 表示句子/样本在降维之前的维度
Train_features3, Test_features3, Train_label3, Test_label3 = [], [], [], []

# ToDo
# 通过models.KeyedVectors加载预训练好的embedding
fast_embedding = models.KeyedVectors.load('./fast_model')
w2v_embedding = models.KeyedVectors.load('./w2v_model')

print("fast_embedding输出词表的个数{},w2v_embedding输出词表的个数{}".format(
    len(fast_embedding.wv.key_to_index), len(w2v_embedding.wv.key_to_index)))

print("取词向量成功")