from sklearn.tree import DecisionTreeRegressor
import numpy as np
import joblib

def get_data(file_loc):
    """
       get data
       :param file_loc: file path
       :return data: ndarray
    """
    f = open(file_loc, 'r')
    data = []
    for line in f:
        new_arr = []
        arr = line.split(' #')[0].split()
        # print(arr)
        score = arr[0]
        q_id = arr[1].split(':')[1]
        new_arr.append(int(round(float(score))))
        new_arr.append(int(q_id))
        arr = arr[2:]
        for el in arr:
            new_arr.append(float(el.split(':')[1]))
        data.append(new_arr)
    f.close()
    return np.array(data)

def group_queries(data):
    """
    Listwise strategy
        :param data: ndarray
        :return dict:{qid: [doc_index1,doc_index2,...]}

    """
    query_indexes = {}
    index = 0
    for record in data:
        query_indexes.setdefault(record[1], [])
        query_indexes[record[1]].append(index)
        index += 1
    return query_indexes


def dcg(scores):
    """
    compute the DCG value based on the given score
    :param scores: a score list of documents
    :return v: DCG value
    """
    v = 0
    for i in range(len(scores)):
        v += (np.power(2, scores[i]) - 1) / np.log2(i+2)  # i+2 is because i starts from 0
    return v

def single_dcg(scores, i, j):
    """
    compute the single dcg that i-th element located j-th position
    :param scores:
    :param i:
    :param j:
    :return:
    """
    return (np.power(2, scores[i]) - 1) / np.log2(j+2)


def idcg(scores):
    """
    compute the IDCG value (best dcg value) based on the given score
    :param scores: a score list of documents
    :return:  IDCG value
    """
    best_scores = sorted(scores)[::-1]
    return dcg(best_scores)


def ndcg(scores):
    """
    compute the NDCG value based on the given score
    :param scores: a score list of documents
    :return:  NDCG value
    """
    return dcg(scores)/idcg(scores)

#
# def delta_ndcg(scores, p, q):
#     """w[i] += rho * rho_complement * delta
#     swap the i-th and j-th doucment, compute the absolute value of NDCG delta
#     :param scores: a score list of documents
#     :param p, q: the swap positions of documents
#     :return: the absolute value of NDCG delta
#     """
#     s2 = scores.copy()  # new score list
#     s2[p], s2[q] = s2[q], s2[p]  # swap
#     return abs(ndcg(s2) - ndcg(scores))


def ndcg_k(scores, k):
    scores_k = scores[:k]
    dcg_k = dcg(scores_k)
    idcg_k = dcg(sorted(scores)[::-1][:k])
    if idcg_k == 0:
        return np.nan
    return dcg_k/idcg_k

def group_by(data, qid_index):
    """

    :param data: input_data
    :param qid_index: the column num where qid locates in input data
    :return: a dict group by qid
    """
    qid_doc_map = {}
    idx = 0
    for record in data:
        qid_doc_map.setdefault(record[qid_index], [])
        qid_doc_map[record[qid_index]].append(idx)
        idx += 1
    # print(qid_doc_map.keys())
    return qid_doc_map


def get_pairs(scores):
    """

    :param scores: given score list of documents for a particular query
    :return: the documents pairs whose firth doc has a higher value than second one.
    """
    pairs = []
    for i in range(len(scores)):
        for j in range(len(scores)):
            if scores[i] > scores[j]:
                pairs.append((i, j))
    return pairs


def compute_lambda(true_scores, temp_scores, order_pairs, qid):
    """

    :param true_scores: the score list of the documents for the qid query
    :param temp_scores: the predict score list of the these documents
    :param order_pairs: the partial oder pairs where first document has higher score than the second one
    :param qid: specific query id
    :return:
        lambdas: changed lambda value for these documents
        w: w value
        qid: query id
    """
    doc_num = len(true_scores)
    lambdas = np.zeros(doc_num)
    w = np.zeros(doc_num)
    IDCG = idcg(true_scores)
    single_dcgs = {}
    for i, j in order_pairs:
        if (i, i) not in single_dcgs:
            single_dcgs[(i, i)] = single_dcg(true_scores, i, i)
        if (j, j) not in single_dcgs:
            single_dcgs[(j, j)] = single_dcg(true_scores, j, j)
        single_dcgs[(i, j)] = single_dcg(true_scores, i, j)
        single_dcgs[(j, i)] = single_dcg(true_scores, j, i)

    for i, j in order_pairs:
        delta = abs(single_dcgs[(i,j)] + single_dcgs[(j,i)] - single_dcgs[(i,i)] -single_dcgs[(j,j)])/IDCG
        rho = 1 / (1 + np.exp(temp_scores[i] - temp_scores[j]))
        lambdas[i] += rho * delta
        lambdas[j] -= rho * delta

        rho_complement = 1.0 - rho
        w[i] += rho * rho_complement * delta
        w[i] -= rho * rho_complement * delta


    return lambdas, w, qid


class LambdaMART():

    def __init__(self, number_of_trees=10, lr = 0.001):
        self.number_of_trees = number_of_trees
        self.lr = lr
        self.trees = []

    def fit(self,data):
        """
        train the model to fit the train dataset
        """
        qid_doc_map = group_by(data, 1)
        query_idx = qid_doc_map.keys()
        # true_scores is a matrix, different rows represent different queries
        true_scores = [data[qid_doc_map[qid], 0] for qid in query_idx]

        order_paris = []
        for scores in true_scores:
            order_paris.append(get_pairs(scores))

        sample_num = len(data)
        predicted_scores = np.zeros(sample_num)
        for k in range(self.number_of_trees):
            print('Tree %d' % k)
            lambdas = np.zeros(sample_num)
            w = np.zeros(sample_num)

            temp_score = [predicted_scores[qid_doc_map[qid]] for qid in query_idx]
            zip_parameters = zip(true_scores, temp_score, order_paris, query_idx)
            for ts, temps, op, qi in zip_parameters:
                sub_lambda, sub_w, qid = compute_lambda(ts, temps, op, qi)
                lambdas[qid_doc_map[qid]] = sub_lambda
                w[qid_doc_map[qid]] = sub_w
            tree = DecisionTreeRegressor(max_depth=50)
            tree.fit(data[:, 2:], lambdas)
            self.trees.append(tree)
            pred = tree.predict(data[:, 2:])
            predicted_scores += self.lr * pred

            # print NDCG
            qid_doc_map = group_by(data, 1)
            ndcg_list = []
            for qid in qid_doc_map.keys():
                subset = qid_doc_map[qid]
                sub_pred_score = predicted_scores[subset]

                # calculate the predicted NDCG
                true_label = data[qid_doc_map[qid], 0]
                topk = len(true_label)
                pred_sort_index = np.argsort(sub_pred_score)[::-1]
                true_label = true_label[pred_sort_index]
                ndcg_val = ndcg_k(true_label, topk)
                ndcg_list.append(ndcg_val)
            print('Epoch:{}, Average NDCG : {}'.format(k, np.nanmean(ndcg_list)))

    def predict(self, data):
        """
        predict the score for each document in testset
        :param data: given testset
        :return:
        """
        qid_doc_map = group_by(data, 1)
        predicted_scores = np.zeros(len(data))
        for qid in qid_doc_map.keys():
            sub_result = np.zeros(len(qid_doc_map[qid]))
            for tree in self.trees:
                sub_result += self.lr * tree.predict(data[qid_doc_map[qid], 2:])
            predicted_scores[qid_doc_map[qid]] = sub_result
        return predicted_scores

    def validate(self, data, k):
        """
        validate the NDCG metric
        :param data: given th testset
        :param k: used to compute the NDCG@k
        :return:
        """
        qid_doc_map = group_by(data, 1)
        ndcg_list = []
        predicted_scores = np.zeros(len(data))
        for qid in qid_doc_map.keys():
            sub_pred_result = np.zeros(len(qid_doc_map[qid]))
            for tree in self.trees:
                sub_pred_result += self.lr * tree.predict(data[qid_doc_map[qid], 2:])
            predicted_scores[qid_doc_map[qid]] = sub_pred_result
            # calculate the predicted NDCG
            true_label = data[qid_doc_map[qid], 0]
            pred_sort_index = np.argsort(sub_pred_result)[::-1]
            true_label = true_label[pred_sort_index]
            ndcg_val = ndcg_k(true_label, k)
            ndcg_list.append(ndcg_val)
        return ndcg_list,predicted_scores

    def save(self,fname):
        """
        save model.

        """
        joblib.dump(self,('%s.joblib' % (fname)),compress=1)



    def load(self,fname):
        """
        load model.

        """
        model = joblib.load(fname)
        return model



if __name__ == '__main__':
    a=[18219.0, 18230.0, 18328.0, 18342.0, 18356.0, 18371.0, 18377.0, 18378.0, 18386.0, 18400.0, 18401.0, 18402.0,
         18410.0, 18411.0, 18429.0, 18437.0, 18438.0, 18450.0, 18457.0, 18458.0, 18464.0, 18468.0, 18470.0, 18479.0,
         18488.0, 18489.0, 18490.0, 18511.0, 18525.0, 18526.0, 18531.0, 18552.0, 18571.0, 18574.0, 18577.0, 18599.0,
         18603.0, 18626.0, 18629.0, 18631.0, 18638.0, 18648.0, 18662.0, 18686.0, 18699.0, 18714.0, 18731.0, 18733.0,
         18738.0, 18765.0, 18767.0, 18774.0, 18806.0, 18822.0, 18826.0, 18838.0, 18844.0, 18853.0, 18889.0, 18910.0,
         18919.0, 18930.0, 18960.0, 18963.0, 18979.0, 18985.0, 18995.0, 18996.0, 19003.0, 19034.0, 19041.0, 19042.0,
         19059.0, 19097.0, 19099.0, 19101.0, 19108.0, 19116.0, 19121.0, 19128.0, 19140.0, 19153.0, 19174.0, 19182.0,
         19186.0, 19193.0, 19216.0, 19236.0, 19248.0, 19254.0, 19276.0, 19349.0, 19352.0, 19353.0, 19356.0, 19364.0,
         19370.0, 19371.0, 19372.0, 19383.0, 19390.0, 19396.0, 19400.0, 19413.0, 19419.0, 19437.0, 19449.0, 19454.0,
         19457.0, 19483.0, 19486.0, 19487.0, 19493.0, 19494.0, 19503.0, 19511.0, 19536.0, 19548.0, 19554.0, 19576.0,
         19586.0, 19595.0, 19602.0, 19603.0, 19633.0, 19681.0, 19682.0, 19720.0, 19731.0, 19736.0, 19737.0, 19756.0,
         19770.0, 19771.0, 19782.0, 19806.0, 19808.0, 19812.0, 19836.0, 19851.0, 19853.0, 19854.0, 19864.0, 19875.0,
         19898.0, 19906.0, 19913.0, 19916.0, 19921.0, 19938.0, 19949.0, 19954.0, 19960.0, 19965.0, 19987.0, 19997.0]
    print(len(a))
    # training_data = np.load('./dataset/train.npy')
    # print(training_data)
    #
    # model = LambdaMART( 5, 0.01)
    # model.fit(training_data)
    #
    # k = 4
    # test_data = np.load('./dataset/test.npy')
    # ndcg = model.validate(test_data, k)
    # print(ndcg)

    training_data = get_data('./train.txt')
    # print(training_data)

    # print(group_queries(training_data))
    # print(group_by(training_data,1))

    model = LambdaMART(training_data, 5, 0.01)
    model.fit()


    k = 5
    test_data = get_data('./test.txt')
    # print(group_queries(test_data))
    # print(group_by(test_data, 1))
    ndcg,predict_scores = model.validate(test_data, k)
    print(ndcg)
    print(len(ndcg))
    print(predict_scores)
    print(len(predict_scores))
