# -*- coding: utf-8 -*-

#有时候需要加
# import findspark
# findspark.init()
import logging
import re
from pyspark import SparkContext,SparkConf,SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF,IDF,VectorAssembler,Word2Vec,Word2VecModel,IDFModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType,IntegerType,ArrayType,FloatType
import jieba
jieba.setLogLevel(jieba.logging.INFO)
# import pyhanlp
from functools import reduce
import pyspark.sql.functions as F
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import CountVectorizerModel
from pyspark.sql.functions import lit
from sklearn.metrics import confusion_matrix,mean_squared_error
import pandas as pd
import numpy as np
import xgboost as xgb
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.linalg import Vectors
from sklearn import metrics
from sklearn.model_selection import train_test_split

def CreateSparkContext():
    sparkconf = SparkConf()\
        .setAppName('xgb_all')\
        .set('spark.ui.showConsoleProgress','false')\
        .set('spark.driver.memory','4g')\
        .set('spark.executor.memory','4g')\
        .set('spark.excutor.cores','4')
    sc = SparkContext(conf = sparkconf)
    print('master:'+sc.master)
    sc.setLogLevel('WARN')

    spark = SparkSession.builder.config(conf=sparkconf).getOrCreate()
    return sc,spark

def load_data(file_path):
    df = spark.read.format('csv')\
        .option('header','true')\
        .option('delimiter',',')\
        .option('inferSchema','true')\
        .load(file_path)
    return df

def label_tran_udf(lab):
    if lab == '洗发水':
        return 0
    elif lab == '书籍':
        return 1
    elif lab == '平板':
        return 2
    elif lab == '计算机':
        return 3
    elif lab == '衣服':
        return 4
    elif lab == '蒙牛':
        return 5
    elif lab == '手机':
        return 6
    elif lab == '水果':
        return 7
    elif lab == '热水器':
        return 8
    else:
        return 9

def set_index_udf(x):
    global idx  # 将idx设置为全局变量
    if x is not None:
        idx += 1
        return index_list[idx - 1]

def remove_punc_udf(line):
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^\u4E00-\u9FA5]")
    line = rule.sub('', line)
    return line

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath,'r',encoding='utf-8').readlines()]
    return stopwords

def seg(x):
    stopwords = stopwordslist('chineseStopWords.txt')
    s = jieba.lcut(x, cut_all=False, HMM=False)
    s = [x for x in s if ((len(x) > 1)&(x not in stopwords))]
    return s

# def seg(x):
#     s = pyhanlp.HanLP.segment(x)
#     s = [x.word for x in s]
#     return s

# def get_by_tfidf(partition):
#
#     for row in partition:
#         TOPK = row.token_count
#         # 找到索引与IDF值并进行排序
#         _dict = list(zip(row.idfFeatures.indices, row.idfFeatures.values))
#         result = _dict[:TOPK]
#         for word_index, tfidf in result:
#             yield row.id,int(word_index), round(float(tfidf), 4)


if __name__ == '__main__':
    sc,spark = CreateSparkContext()
    text_df = load_data('online_shopping_10_cats.csv')
    # text_df.printSchema()
    # text_df.show(5)

    #数据集有多大
    # print(text_df.count())

    #缺失数据有多少
    df_agg = text_df.agg(*[F.count(F.when(F.isnull(c), c)).alias(c) for c in text_df.columns])
    # df_agg.show(5)
    # print(df_agg.count())

    #统计缺失数据
    df_agg_col = reduce(lambda a, b: a.union(b),
                        (df_agg.select(F.lit(c).alias('Column_Name'), F.col(c).alias('Null_Count')) for c in
                         df_agg.columns))
    # df_agg_col.show(5)

    # 若存在缺失值，可以进行直接删除缺失值所在的行，存下其余的行
    # print("删除缺失数据之前的数据量为：", text_df.count())
    text_df = text_df.dropna()
    # print("删除缺失数据之后的数据量为：", text_df.count())

    # 已经无缺失了
    # 查看cat列每个类的名称
    # text_df.select('cat').distinct().show()
    # print(text_df.select('cat').distinct().count())


    # 对cat列将文本数据转换为数字
    label_tran_udf_df = udf(label_tran_udf, IntegerType())
    text_df = text_df.withColumn('cat_class', label_tran_udf_df(text_df.cat))
    # text_df.show(5)

    #先添加一列为1的常数列
    text_df = text_df.withColumn('constant',lit(1))

    #添加一列索引列
    index_list = [x for x in range(0, text_df.count())]  # 构造一个列表存储索引值，用生成器会出错
    idx = 0
    index = udf(set_index_udf, IntegerType())
    # udf的注册，这里需要定义其返回值类型
    text_df = text_df.withColumn('id', index(text_df.constant))


    # 对文本数据进行处理
    # 清除文本中的非汉字
    remove_punc_udf_df = udf(remove_punc_udf, StringType())
    text_df = text_df.withColumn('text_remove_punc', remove_punc_udf_df(text_df.review))
    # text_df.show(5)

    #去除停用词和切词，使用jieba自带词典
    seg_udf = udf(seg,ArrayType(StringType()))
    text_df = text_df.withColumn('text_token',seg_udf(text_df.text_remove_punc))
    # text_df.show(5)

    #自己加的计算每篇文章所含单词数
    len_udf = udf(lambda s:len(s),IntegerType())
    text_df = text_df.withColumn('token_count',len_udf(text_df.text_token))

    #最终选择的列：
    text_df = text_df.select(['id','text_token','token_count','cat_class'])
    print('在tfidf之前的最终数据形式：')
    text_df.show()

    #在这里我们仅选择前50条数据进行处理
    # text_df = text_df.filter('id<1000')
    #若是有特定领域的新词典（自定义词典）需要加载进来
    # def seg(x):
    #     if not jieba.dt.initialized:
    #         jieba.load_userdict('词典的具体格式，可以有多种')
    #         s = jieba.lcut(x, cut_all=False, HMM=False)
    #         s = [x for x in s if len(x) > 1]
    #         return s
    # seg_df = udf(seg, ArrayType(StringType()))
    # 数据集中含有列名为：keyword 这一列，这一列含有的数据全部是中文文本数据
    # text_df = text_df.withColumn('token_text', seg_df(text_df.keyword))
    # text_df.show()

    # #tfidf训练分词数据
    #hashingtf
    hashing_vec = HashingTF(inputCol='text_token',outputCol='tf')
    hashing_df = hashing_vec.transform(text_df)
    #idf
    idf_vec = IDF(inputCol='tf',outputCol='idf')
    idf_df = idf_vec.fit(hashing_df).transform(hashing_df)
    idf_df.show()

    #集成向量
    va = VectorAssembler(inputCols=['idf','token_count'],outputCol='feature')
    idf_df = va.transform(idf_df)
    idf_df.show()



    idf_df_array = np.array(idf_df.select('feature').collect())
    #idf_df_array = np.array(idf_df.select('feature').toPandas())

    data = idf_df_array.reshape(idf_df.count(), -1)

    label = np.array(idf_df.select('cat_class').collect()).reshape(-1)

    train_x, test_x, train_y, test_y = train_test_split(data, label, random_state=0)

    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x,label=test_y)

    # params = {'booster': 'gbtree',
    #           'objective': 'multi:softmax',
    #           'num_class': 10,
    #           'eval_metric': 'mlogloss',
    #           'max_depth': 4,
    #           'lambda': 10,
    #           'subsample': 0.7,
    #           'colsample_bytree': 0.7,
    #           'min_child_weight': 3,
    #           'eta': 0.025,
    #           'seed': 0,
    #           'nthread': 4
    #           }

    watchlist = [(dtrain, 'train'),(dtest,'test')]
    # bst = xgb.train(params, dtrain, num_boost_round=5, evals=watchlist)
    # 输出概率
    # ypred = bst.predict(dtest)
    # print('predicting, classification error=%f' % (
    #             sum(int(ypred[i]) != test_y[i] for i in range(len(test_y))) / float(len(test_y))))

    # probabilities
    # do the same thing again, but output probabilities
    params_1 = {'booster': 'gbtree',
              'objective': 'multi:softprob',
              'num_class': 10,
              'eval_metric': 'mlogloss',
              'max_depth': 4,
              'lambda': 10,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'min_child_weight': 3,
              'eta': 0.025,
              'seed': 0,
              'nthread': 4
              }

    bst_1 = xgb.train(params_1, dtrain, num_boost_round=5, evals=watchlist)

    # Note: this convention has been changed since xgboost-unity
    # get prediction, this is in 1D array, need reshape to (ndata, nclass)
    yprob = bst_1.predict(dtest).reshape(test_y.shape[0], 10)
    # 从预测的10组中选择最大的概率进行输出
    ylabel = np.argmax(yprob, axis=1)  # return the index of the biggest pro

    # print('predicting, classification error=%f' % (
    #             sum(int(ylabel[i]) != test_y[i] for i in range(len(test_y))) / float(len(test_y))))
    # 最小二乘方差
    # mse2 = mean_squared_error(test_y, ylabel)

    # print(mse2)

    print(confusion_matrix(test_y, ylabel))

    print('ACC: %.4f' % metrics.accuracy_score(test_y, ylabel))
    print('Precesion: %.4f' % metrics.precision_score(test_y, ylabel,average='weighted'))
    print('Recall: %.4f' % metrics.recall_score(test_y, ylabel,average='weighted'))
    print('F1-score: %.4f' % metrics.f1_score(test_y, ylabel,average='weighted'))
    print('AUC_macro_ovo: %.4f' % metrics.roc_auc_score(test_y, yprob,average='macro',multi_class='ovo'))
    print('AUC_macro_ovr: %.4f' % metrics.roc_auc_score(test_y, yprob,average='macro',multi_class='ovr'))
    print('AUC_micro_ovo: %.4f' % metrics.roc_auc_score(test_y, yprob,average='weighted',multi_class='ovo'))
    print('AUC_micro_ovr: %.4f' % metrics.roc_auc_score(test_y, yprob,average='weighted',multi_class='ovr'))


    # text_df_all.coalesce(1).write.option('header','true').csv('pyspark_new_online_shopping.csv')
    # text_df_all = text_df_all.rdd.map(extract).toDF(["id","cat_class"])  # Vector values will be named _2, _3, ...


    # text_df_all.coalesce(1).write.option('header','true').csv('pyspark_new_online_shopping_1.csv')















    # 多分类：对标签列 Label 进行统计,是否存在不平衡问题在，答案是肯定的
    # print(tf_idf_df.select('cat_class').distinct().count())
    #
    #
    # # 编写自定义udf函数计算分好词的每句话（每一行），或者理解为一篇短文档的词长度
    # len_udf = udf(lambda s: len(s), IntegerType())
    # tf_idf_df = tf_idf_df.withColumn('token_count', len_udf(tf_idf_df.text_remove_punc))
    #
    # # 特征集成？
    # va = VectorAssembler(inputCols=['tf_idf_feature', 'token_count'], outputCol='feature_vec')
    # tf_idf_df = va.transform(tf_idf_df)
    # tf_idf_df.printSchema()
    #
    # 建模
    # 划分训练集和测试集
    # train_df, test_df = tf_idf_df.randomSplit([0.8, 0.2])
    #
    # clf_lr = LogisticRegression(featuresCol='feature_vec', labelCol='cat_class').fit(train_df)
    # predictions = clf_lr.transform(test_df)
    # predictions.show()
    #
    # evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='cat_class', metricName='f1')
    # f1 = evaluator.evaluate(predictions)
    # print('使用tf-idf进行处理的f1为', f1)


    # cm_mul = confusion_matrix(predictions.toPandas()['cat_class'].tolist(),predictions.toPandas()['prediction'].tolist())
    # print(cm_mul)
    # print(cm_mul.shape)
    # print(type(cm_mul))
    # #关于多分类，对于每个类进行统计与预测
    # f = open('2021_4_15.txt','w')
    # p_i = []
    # r_i = []
    # f1_i = []
    # for i in range(cm_mul.shape[0]):
    #     p_i.append(cm_mul[i,i]/sum(cm_mul[:,i].tolist()))
    #     r_i.append(cm_mul[i,i]/sum(cm_mul[i,:].tolist()))
    #
    # for i in range(len(p_i)):
    #     f1_i.append(2*p_i[i]*r_i[i]/(p_i[i]+r_i[i]))
    #     f.write(str(i))
    #     f.write(str(f1_i[i]))
    # f.close()


