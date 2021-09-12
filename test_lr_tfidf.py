# -*- coding: utf-8 -*-

import re
import jieba
import pandas as pd
import pyspark.sql.functions as F
from functools import reduce
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark import SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import IDF
from pyspark.ml.feature import IDFModel
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import Word2VecModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import ArrayType
from pyspark.sql.types import FloatType
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import CountVectorizerModel
from pyspark.sql.functions import lit
from sklearn.metrics import confusion_matrix


def create_spark_context():
    sparkconf = SparkConf()\
        .setAppName('xxx')\
        .set('spark.ui.showConsoleProgress','false')\
        .set('spark.driver.memory','4g')\
        .set('spark.executor.memory','4g')\
        .set('spark.excutor.cores','4')
    sc = SparkContext(conf = sparkconf)
    print('master:' + sc.master)
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

# 对cat列将文本数据转换为数字
def label_tran_udf(lab):
    if lab == '洗发水':
        return 1.0
    elif lab == '书籍':
        return 2.0
    elif lab == '平板':
        return 3.0
    elif lab == '计算机':
        return 4.0
    elif lab == '衣服':
        return 5.0
    elif lab == '蒙牛':
        return 6.0
    elif lab == '手机':
        return 7.0
    elif lab == '水果':
        return 8.0
    elif lab == '热水器':
        return 9.0
    else:
        return 10.0

# 定义一个函数
def set_index_udf(x):
    global idx  # 将idx设置为全局变量
    if x is not None:
        idx += 1
        return index_list[idx - 1]

# 对文本数据进行处理
# 清除文本中的非汉字
def remove_punc_udf(line):
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^\u4E00-\u9FA5]")
    line = rule.sub('', line)
    return line

#切词，使用jieba自带词典
def seg(x):
    s = jieba.lcut(x, cut_all=False, HMM=False)
    s = [x for x in s if len(x) > 1]
    return s

#若是有特定领域的新词典（自定义词典）需要加载进来
# def seg(x):
#     if not jieba.dt.initialized:
#         jieba.load_userdict('词典的具体格式，可以有多种')
#         s = jieba.lcut(x, cut_all=False, HMM=False)
#         s = [x for x in s if len(x) > 1]
#         return s

if __name__ == '__main__':
    sc, spark = create_spark_context()
    text_df = load_data('online_shopping_10_cats.csv')
    text_df.printSchema()
    print(text_df.count())
    text_df.describe()
    text_df.cache()
    text_df.show()
    print(text_df.count())
    df_agg = text_df.agg(*[F.count(F.when(F.isnull(c), c)).alias(c) for c in text_df.columns])
    df_agg.show()
    print(df_agg.count())
    df_agg_col = reduce(lambda a, b: a.union(b),
                        (df_agg.select(F.lit(c).alias('Column_Name'), F.col(c).alias('Null_Count')) for c in
                         df_agg.columns))

    df_agg_col.show()
    # 若存在缺失值，可以进行直接删除缺失值所在的行，存下其余的行
    print("删除缺失数据之前的数据量为：", text_df.count())
    text_df = text_df.dropna()
    print("删除缺失数据之后的数据量为：", text_df.count())

    # 已经无缺失了
    # 查看cat列每个类的名称
    text_df.select('cat').distinct().show()
    print(text_df.select('cat').distinct().count())

    label_tran_udf_df = udf(label_tran_udf, FloatType())
    text_df = text_df.withColumn('cat_class', label_tran_udf_df(text_df.cat))
    text_df.show()

    #先添加一列为1的常数列
    text_df = text_df.withColumn('constant',lit(1))

    #添加一列索引列
    index_list = [x for x in range(0, text_df.count())]  # 构造一个列表存储索引值，用生成器会出错
    idx = 0

    index = udf(set_index_udf, IntegerType())  # udf的注册，这里需要定义其返回值类型
    text_df = text_df.withColumn('id', index(text_df.constant))

    remove_punc_udf_df = udf(remove_punc_udf, StringType())
    text_df = text_df.withColumn('text_remove_punc', remove_punc_udf_df(text_df.review))
    text_df.show()

    seg_udf = udf(seg,ArrayType(StringType()))
    text_df = text_df.withColumn('text_token',seg_udf(text_df.text_remove_punc))
    text_df.show()

    # seg_df = udf(seg, ArrayType(StringType()))
    # 数据集中含有列名为：keyword 这一列，这一列含有的数据全部是中文文本数据
    # text_df = text_df.withColumn('token_text', seg_df(text_df.keyword))
    # text_df.show()


    # 关键词获取(tfidf)
    # tdidf
    # 词频，即tf
    # vocabSize是总词汇的大小，minDF是文本中出现的最少次数, vocabSize=200 * 10000, minDF=1.0
    cv = CountVectorizer(inputCol="text_token", outputCol="countFeatures")
    # 训练词频统计模型
    cv_model = cv.fit(text_df)
    cv_model.write().overwrite().save("model/CV.model")
    cv_model = CountVectorizerModel.load("model/CV.model")
    # 得出词频向量结果
    cv_result = cv_model.transform(text_df)
    cv_result.show()
    cv_result.printSchema()

    # idf
    idf = IDF(inputCol="countFeatures", outputCol="idfFeatures")
    idf_model = idf.fit(cv_result)
    idf_model.write().overwrite().save("model/IDF.model")

    # tf-idf
    idf_model = IDFModel.load("model/IDF.model")
    tfidf_result = idf_model.transform(cv_result)
    tfidf_result.show()
    tfidf_result.printSchema()

    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

    # # 多分类：对标签列 Label 进行统计,是否存在不平衡问题在，答案是肯定的
    # print(tfidf_result.select('cat_class').distinct().count())

    # 编写自定义udf函数计算分好词的每句话（每一行），或者理解为一篇短文档的词长度
    len_udf = udf(lambda s: len(s), IntegerType())
    tf_idf_df = tfidf_result.withColumn('token_count', len_udf(tfidf_result.text_remove_punc))

    # 特征集成
    va = VectorAssembler(inputCols=['idfFeatures', 'token_count'], outputCol='feature_vec')
    tf_idf_df = va.transform(tf_idf_df)
    tf_idf_df.printSchema()

    # 建模
    # 划分训练集和测试集
    train_df, test_df = tf_idf_df.randomSplit([0.8, 0.2])

    clf_lr = LogisticRegression(featuresCol='feature_vec', labelCol='cat_class').fit(train_df)
    predictions = clf_lr.transform(test_df)
    predictions.show()
    predictions.printSchema()

    evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='cat_class', metricName='f1')
    f1 = evaluator.evaluate(predictions)
    print('使用tf-idf进行处理的f1为', f1)


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














