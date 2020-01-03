import sys

sys.path.extend(["../../", "../", "./"])
import warnings

warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
import gensim
import re
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
import time
import argparse
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from gensim.models import word2vec


# ======================================================================================================================
# 将fatsa文件切分成单词默认为kmer切分
# ======================================================================================================================
# kmer切分 :b = [string[i:i + 3] for i in range(len(string)) if i < len(string) - 2]
# 普通分词 : b = re.findall(r'.{3}', string)
def save_wordfile(fastafile, wordfile, splite, kmer):
    f = open(fastafile)
    f1 = open(wordfile, "w")
    k = kmer - 1
    documents = f.readlines()
    string = ""
    flag = 0
    for document in documents:
        if document.startswith(">") and flag == 0:
            flag = 1
            continue
        elif document.startswith(">") and flag == 1:
            if splite == 0:
                b = [string[i:i + kmer] for i in range(len(string)) if i < len(string) - k]
            else:
                b = re.findall(r'.{3}', string)
            word = " ".join(b)
            f1.write(word)
            f1.write("\n")
            string = ""
        else:
            string += document
            string = string.strip()
    if splite == 0:
        b = [string[i:i + kmer] for i in range(len(string)) if i < len(string) - k]
    else:
        b = re.findall(r'.{kmer}', string)
    word = " ".join(b)
    f1.write(word)
    f1.write("\n")
    print("words have been saved in file {}！\n".format(wordfile))
    f1.close()
    f.close()


def splite_word(trainfasta_file, trainword_file, kmer, testfasta_file, testword_file, splite, flag):
    train_file = trainfasta_file
    train_wordfile = trainword_file
    test_file = testfasta_file
    test_wordfile = testword_file

    # train set transform to word
    save_wordfile(train_file, train_wordfile, splite, kmer)
    # testing set transform to word
    if flag:
        save_wordfile(test_file, test_wordfile, splite, kmer)


# ======================================================================================================================
# 训练词向量并将文件转化为csv文件
# ======================================================================================================================
def save_csv(word_file, model, csv_file, b):
    wv = model.wv
    vocab_list = wv.index2word
    feature = []
    outputfile = csv_file
    with open(word_file) as f:
        # l = []
        words = f.readlines()
        for word in words:
            l = []
            cc = word.split(" ")
            for i in cc:
                i = i.rstrip()
                if i not in vocab_list:
                    flag = [b] * 100
                else:
                    flag = model[i]
                l.append(flag)
            word_vec = np.array(l)
            feature.append(np.mean(word_vec, axis=0))
        pd.DataFrame(feature).to_csv(outputfile, header=None, index=False)

    print("csv have been saved in file {}！\n".format(outputfile))


def tocsv(trainword_file, testword_file, sg, hs, window, size, model_name, traincsv, testcsv, b, flag, iter, spmodel):
    if spmodel:
        print("loading model")
        model = gensim.models.KeyedVectors.load_word2vec_format(spmodel, binary=False)
    else:
        sentences = word2vec.LineSentence(trainword_file)
        model = word2vec.Word2Vec(sentences, sg=sg, hs=hs, min_count=1, window=window, size=size, iter=iter)
        model.wv.save_word2vec_format(model_name, binary=False)

    save_csv(trainword_file, model, traincsv, b)

    if flag:
        save_csv(testword_file, model, testcsv, b)


# ======================================================================================================================
# svm
# ======================================================================================================================

def svm(traincsv, trainpos, trainneg, testcsv, testpos, testneg, cv, n_job, mms, ss, flag, grad):
    cv = cv
    cpu_num = n_job
    svc = SVC(probability=True)

    X = pd.read_csv(traincsv, header=None, sep=",")
    y = np.array([0] * trainpos + [1] * trainneg)

    if flag:
        X1 = pd.read_csv(testcsv, header=None, sep=",")
        y1 = np.array([0] * testpos + [1] * testneg)

    if mms:
        print("MinMaxScaler")
        minMax = MinMaxScaler()
        minMax.fit(X)
        X = minMax.transform(X)
        if flag:
            X1 = minMax.transform(X1)

    if ss:
        print("StandardScaler")
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        if flag:
            X1 = scaler.transform(X1)

    # 网格搜索
    def get_bestparameter(X, y):

        a = [2 ** x for x in range(-2, 5)]
        b = [2 ** x for x in range(-5, 2)]
        parameters = [
            {
                'C': a,
                'gamma': b,
                'kernel': ['rbf']
            },
            {
                'C': a,
                'kernel': ['linear']
            }
        ]
        clf = GridSearchCV(svc, parameters, cv=cv, scoring='accuracy', n_jobs=cpu_num)
        clf.fit(X, y)

        print("Best parameters set found on development set:")
        print(clf.best_params_)
        print(clf.best_score_)

        return clf

    # clf = get_bestparameter(X, y)

    if flag:
        print("------------------supplied the test set----------------------------")
        if grad:
            clf = get_bestparameter(X, y)
        else:
            clf = SVC(C=0.5, gamma=0.05)
            clf.fit(X, y)
        pre = clf.predict(X1)
        print("ACC:{}".format(metrics.accuracy_score(y1, pre)))
        print("MCC:{}".format(metrics.matthews_corrcoef(y1, pre)))
        print(classification_report(y1, pre))
        print("confusion matrix\n")
        print(pd.crosstab(pd.Series(y1, name='Actual'), pd.Series(pre, name='Predicted')))
    
    else:
        print("------------------------cv--------------------------")
        label = [0, 1]
        if grad:
            clf = get_bestparameter(X, y)
            p = clf.best_params_
            if clf.best_params_["kernel"] == "rbf":
                clf = SVC(C=p["C"], kernel=p["kernel"], gamma=p["gamma"], probability=True)
            else:
                clf = SVC(C=p["C"], kernel=p["kernel"], probability=True)
        else:
            clf = SVC(C=0.5, gamma=0.05, probability=True)

        predicted = cross_val_predict(clf, X, y, cv=cv, n_jobs=cpu_num)
        y_predict_prob = cross_val_predict(clf, X, y, cv=cv, n_jobs=cpu_num, method='predict_proba')
        ROC_AUC_area = metrics.roc_auc_score(y, y_predict_prob[:, 1])

        print("AUC:{}".format(ROC_AUC_area))
        print("ACC:{}".format(metrics.accuracy_score(y, predicted)))
        print("MCC:{}\n".format(metrics.matthews_corrcoef(y, predicted)))
        print(classification_report(y, predicted))
        print("confusion matrix\n")
        print(pd.crosstab(pd.Series(y, name='Actual'), pd.Series(predicted, name='Predicted')))


# ======================================================================================================================
# 主函数
# ======================================================================================================================
def main():
    parser = argparse.ArgumentParser()
    # parameter of train set
    parser.add_argument('-trainfasta', required=True, help="trainfasta file name")
    parser.add_argument('-trainword', default="trainword.txt", help="file name of train set")
    parser.add_argument('-trainpos', required=True, type=int, help="trainpos")
    parser.add_argument('-trainneg', required=True, type=int, help="trainneg")
    parser.add_argument('-traincsv', default="train.csv", help="csv file name of train set")
    # parameter of word2vec
    parser.add_argument('-b', default=0, help="Fill in the vector")
    parser.add_argument('-sg', type=int, default=1, help="")
    parser.add_argument('-iter', type=int, default=5, help="")
    parser.add_argument('-hs', type=int, default=0, help="")
    parser.add_argument('-spmodel', help="spmodel")
    parser.add_argument('-window_size', type=int, default=20, help="window size")
    parser.add_argument('-model', default="model.model", help="embedding model")
    parser.add_argument('-hidden_size', type=int, default=100, help="The dimension of word")
    # parameter of testing set
    parser.add_argument('-testfasta', help="testfasta file name")
    parser.add_argument('-testword', default="testword.txt", help="file name of testing set")
    parser.add_argument('-testpos', type=int, help="testpos")
    parser.add_argument('-testneg', type=int, help="testneg")
    parser.add_argument('-testcsv', default="test.csv", help="csv file name of testing set")
    # svm
    parser.add_argument('-mms', type=bool, default=False, help="minmaxscaler")
    parser.add_argument('-ss', type=bool, default=False, help="StandardScaler")
    parser.add_argument('-cv', type=int, default=10, help="cross validation")
    parser.add_argument('-n_job', '-n', default=-1, help="num of thread")
    parser.add_argument('-grad', type=bool, default=False, help="grad")
    # splite
    parser.add_argument('-kmer', '-k', type=int, default=3, help="k-mer: k size")
    parser.add_argument('-splite', '-s', type=int, default=0, help="kmer splite(0) or normal splite(1)")

    args = parser.parse_args()
    print(args)
    flag = False
    if args.testfasta:
        flag = True
    if args.splite == 0:
        print("kmer splite !")
    else:
        print("normal splite !")

    start_time = time.time()

    splite_word(args.trainfasta, args.trainword, args.kmer, args.testfasta, args.testword, args.splite, flag)

    tocsv(args.trainword, args.testword, args.sg, args.hs, args.window_size, args.hidden_size, args.model,
          args.traincsv, args.testcsv, args.b, flag, args.iter, args.spmodel)

    svm(args.traincsv, args.trainpos, args.trainneg, args.testcsv, args.testpos, args.testneg, args.cv, args.n_job,
        args.mms, args.ss, flag, args.grad)

    end_time = time.time()
    print("end ............................")
    print("Time consuming：{}s\n".format(end_time - start_time))


if __name__ == '__main__':
    main()
