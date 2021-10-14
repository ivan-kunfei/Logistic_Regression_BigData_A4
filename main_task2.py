import re
import sys
import numpy as np
from pyspark import SparkContext
from pyspark.storagelevel import StorageLevel
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: wordcount <file> <output> ", file=sys.stderr)
        exit(-1)

    print("===== Task 1 - Data Preparation and Model Training =====")
    sc = SparkContext(appName="LogisticRegression")
    train_dir = sys.argv[1]  # "SmallTrainingData.txt"
    test_dir = sys.argv[2]  # "SmallTrainingData.txt"
    d_corpus = sc.textFile(train_dir, 1)  # min partition=1
    valid_lines = d_corpus.filter(lambda x: 'id="' in x and '" url=' in x and '">' in x)
    top_size = 20000
    d_keyAndText = valid_lines.map(lambda x: (x[x.index('id="') + 4: x.index('" url=')], x[x.index('">') + 2:][:-6]))
    regex = re.compile('[^a-zA-Z]')
    d_keyAndListOfWords = d_keyAndText.map(lambda x: (x[0], regex.sub(' ', x[1]).lower().split()))
    all_words = d_keyAndListOfWords.flatMap(lambda x: [(word, 1) for word in x[1]])
    all_counts = all_words.reduceByKey(lambda x, y: x + y)
    top_words = all_counts.top(top_size, key=lambda x: x[1])
    top_words_k = sc.parallelize(range(top_size))
    dictionary = top_words_k.map(lambda x: (top_words[x][0], x))
    dictionary.persist(storageLevel=StorageLevel(True, False, False, False))
    all_words_with_docid = d_keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))
    # Now join and link them, to get a set of ("word", (dictionaryPos, doc type)) pairs
    all_dictionary_words = dictionary.join(all_words_with_docid)
    just_doc_and_pos = all_dictionary_words.map(lambda x: (x[1][1], x[1][0]))
    all_dictionary_words_in_each_doc = just_doc_and_pos.groupByKey()


    def build_array(list_of_indices):
        return_val = np.zeros(top_size)
        for index in list_of_indices:
            return_val[index] = return_val[index] + 1
        my_sum = np.sum(return_val)
        return_val = np.divide(return_val, my_sum)
        return return_val


    def get_category(docid):
        if docid[0:2] == "AU":
            return 1
        else:
            return 0


    #  Train TF Array
    all_docs_as_numpy_arrays = all_dictionary_words_in_each_doc.map(lambda x: (get_category(x[0]), build_array(x[1])))
    all_docs_as_numpy_arrays.persist(storageLevel=StorageLevel(True, False, False, False))


    def build_zero_one_array(list_of_indices):
        return_val = np.zeros(top_size)
        for index in list_of_indices:
            return_val[index] = 1
        return return_val


    zero_or_one = all_dictionary_words_in_each_doc.mapValues(build_zero_one_array).map(lambda x: (1, x[1]))
    number_of_docs, df_array = zero_or_one.reduce(lambda x1, x2: (x1[0] + x2[0], np.add(x1[1], x2[1])))
    idf_array = np.log(number_of_docs / df_array)
    tf_idf = all_docs_as_numpy_arrays.map(lambda x: (x[0], np.multiply(x[1], idf_array)))
    tf_idf.cache()

    a, b, c, d, e = False, False, False, False, False
    for i in range(top_size):
        word = top_words[i][0]
        if a and b and c and d and e:
            break
        if word == "applicant":
            a = i
            continue
        if word == "and":
            b = i
            continue
        if word == "attack":
            c = i
            continue
        if word == "protein":
            d = i
            continue
        if word == "court":
            e = i
    five_words_index = (a, b, c, d, e)


    def get_tf(tf_array):
        result = []
        for index in five_words_index:
            result.append(tf_array[index])
        result.append(1)
        return np.array(result)


    sum_tf = all_docs_as_numpy_arrays.map(lambda x: (x[0], get_tf(x[1])))
    sum_tf = sum_tf.reduceByKey(np.add).collect()
    court_avg_tf = 0
    wiki_avg_tf = 0
    for each in sum_tf:
        if each[0] == 1:
            court_avg_tf = each[1]
        else:
            wiki_avg_tf = each[1]

    court_avg_tf = court_avg_tf[0:5] / court_avg_tf[-1]
    wiki_avg_tf = wiki_avg_tf[0:5] / wiki_avg_tf[-1]
    print("TF value of 5 words for court: {}".format(court_avg_tf))
    print("TF value of 5 words for wiki: {}".format(wiki_avg_tf))

    test_d_corpus = sc.textFile(test_dir, 1)  # min partition=1
    test_valid_lines = test_d_corpus.filter(lambda x: 'id="' in x and '" url=' in x and '">' in x)
    test_text = test_valid_lines.map(lambda x: (x[x.index('id="') + 4: x.index('" url=')], x[x.index('">') + 2:][:-6]))
    test_words = test_text.map(lambda x: (x[0], regex.sub(' ', x[1]).lower().split()))
    test_words_id = test_words.flatMap(lambda x: ((j, x[0]) for j in x[1]))
    test_id_pos = dictionary.join(test_words_id).map(lambda x: (x[1][1], x[1][0]))
    test_tf_array = test_id_pos.groupByKey().map(lambda x: (get_category(x[0]), build_array(x[1])))
    test_tf_idf = test_tf_array.map(lambda x: (x[0], np.multiply(x[1], idf_array)))
    test_tf_idf.persist(storageLevel=StorageLevel(True, False, False, False))

    learning_rate = 0.001
    lbd = 3
    initial_weights = np.random.randn(top_size)
    iteration_num = 80
    old_cost = None


    def train(input_val, weights):
        x = input_val[1]
        y = input_val[0]
        theta = np.dot(x, weights)
        temp = np.exp(theta)
        llh = y * theta - np.log(1 + temp)
        regularization = np.sum(np.square(weights))
        costs = -llh + lbd * regularization
        gradient = - x * y + x * (temp / (1 + temp)) + 2 * lbd * weights
        return gradient, costs, 1


    for i in range(iteration_num):
        sample = sc.parallelize(tf_idf.takeSample(True, 1000))
        sum_gradients, sum_costs, size = sample.map(lambda x: train(x, initial_weights)).reduce(
            lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2]))
        gradients = sum_gradients / size
        cost = sum_costs / size
        initial_weights = initial_weights - learning_rate * gradients
        if old_cost is None:
            old_cost = cost
        else:
            if cost <= old_cost:
                learning_rate *= 1.05
            else:
                learning_rate *= 0.5
            old_cost = cost
        print("Epoch {}, cost: {}".format(i + 1, cost))
        print("weights: {}".format(initial_weights))

    sort_index = initial_weights.argsort()
    top_five_words = []
    for i in range(-1, -6, -1):
        word_index = sort_index[i]
        word = top_words[i][0]
        top_five_words.append(word)
    print("Top 5 words related with court case: {}".format(top_five_words))


    def predict(input_array):
        result = np.dot(input_array, initial_weights)
        if result >= 0:
            return 1
        else:
            return 0


    def evaluate(input_result):
        true = input_result[0]
        predicted = input_result[1]
        if true == 1 and predicted == 1:
            return "TP", 1
        if true == 1 and predicted == 0:
            return "FN", 1
        if true == 0 and predicted == 1:
            return "FP", 1
        if true == 0 and predicted == 0:
            return "TN", 1


    print("===== Task 2 - Evaluation of the learned Model =====")
    final_result = test_tf_idf.map(lambda x: (x[0], predict(x[1]))).map(evaluate).reduceByKey(
        lambda x, y: x + y).collect()
    TP, FN, FP, TN = 1.0, 1.0, 1.0, 1.0
    for each in final_result:
        if each[0] == "TP" and each[1] > 0:
            TP = float(each[1])
            continue
        if each[0] == "FN" and each[1] > 0:
            FN = float(each[1])
            continue
        if each[0] == "FP" and each[1] > 0:
            FP = float(each[1])
            continue
        if each[0] == "TN" and each[1] > 0:
            TN = float(each[1])
            continue

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    print("TP: {}, FN: {}, FP: {}, TN: {}".format(TP, FN, FP, TN))
    print("precision: {}, recall: {}".format(precision, recall))
    f1_score = 2.0 * precision * recall / (precision + recall)
    print("f1_score : {}".format(f1_score))

    print("===== Task 3 - Using Spark Machine Learning Library =====")


    def parsePoint(values):
        return LabeledPoint(values[0], values[1])


    train_data = tf_idf.map(parsePoint)
    test_data = test_tf_idf.map(parsePoint)
    model = LogisticRegressionWithLBFGS.train(data=train_data, iterations=100, regType="l2")
    labelsAndPreds = test_data.map(lambda p: (p.label, model.predict(p.features)))
    final_result = labelsAndPreds.map(evaluate).reduceByKey(lambda x, y: x + y).collect()
    TP, FN, FP, TN = 1.0, 1.0, 1.0, 1.0
    for each in final_result:
        if each[0] == "TP" and each[1] > 0:
            TP = float(each[1])
            continue
        if each[0] == "FN" and each[1] > 0:
            FN = float(each[1])
            continue
        if each[0] == "FP" and each[1] > 0:
            FP = float(each[1])
            continue
        if each[0] == "TN" and each[1] > 0:
            TN = float(each[1])
            continue

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    print("TP: {}, FN: {}, FP: {}, TN: {}".format(TP, FN, FP, TN))
    print("precision: {}, recall: {}".format(precision, recall))
    f1_score = 2.0 * precision * recall / (precision + recall)
    print("f1_score : {}".format(f1_score))
