import sys

sys.path.extend(["../../", "../", "./"])
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
import gensim
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
from sklearn.externals import joblib


# 将fatsa文件切分成单词默认为kmer切分
def save_wordfile(fastafile, splite, kmer):
    train_words = []
    for i in fastafile:
        f = open(i)
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
                    b = [string[i:i + kmer] for i in range(0, len(string), kmer) if i < len(string) - k]

                train_words.append(b)
                string = ""
            else:
                string += document
                string = string.strip()
        if splite == 0:
            b = [string[i:i + kmer] for i in range(len(string)) if i < len(string) - k]
        else:
            b = [string[i:i + kmer] for i in range(0, len(string), kmer) if i < len(string) - k]
        train_words.append(b)
    f.close()
    return train_words


def splite_word(trainfasta_file, kmer, splite):
    train_file = trainfasta_file

    # train set transform to word
    word = save_wordfile(train_file, splite, kmer)

    return word


# 训练词向量并将文件转化为csv文件
def save_csv(words, model, b):
    wv = model.wv
    vocab_list = wv.index2word
    feature = []

    for word in words:
        l = []
        for i in word:
            i = i.strip()
            if i not in vocab_list:
                flag = [b] * 100
            else:
                flag = model[i]
            l.append(flag)
        word_vec = np.array(l)
        feature.append(np.mean(word_vec, axis=0))
    return np.array(feature)


def tocsv(train_word, sg, hs, window, size, model_name, b, iter1, spmodel):

    if spmodel:
        print("loading model ......")
        model = gensim.models.KeyedVectors.load_word2vec_format(spmodel, binary=False)
    else:
        model = word2vec.Word2Vec(train_word, iter=iter1, sg=sg, hs=hs, min_count=1, window=window, size=size)
        model.wv.save_word2vec_format(model_name, binary=False)

    csv = save_csv(train_word, model, b)

    return csv


# svm
def svm(traincsv, train_y, cv, n_job, mms, ss, grad, model):
    cv = cv
    cpu_num = n_job
    svc = SVC(probability=True)

    X = traincsv
    y = train_y

    if mms:
        print("MinMaxScaler")
        minMax = MinMaxScaler()
        minMax.fit(X)
        X = minMax.transform(X)

    if ss:
        print("StandardScaler")
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

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

    if grad:
        clf = get_bestparameter(X, y)
        p = clf.best_params_
        if clf.best_params_["kernel"] == "rbf":
            clf = SVC(C=p["C"], kernel=p["kernel"], gamma=p["gamma"], probability=True)
        else:
            clf = SVC(C=p["C"], kernel=p["kernel"], probability=True)
    else:
        clf = SVC(C=0.5, gamma=0.05, probability=True)
    if cv:
        print("------------------------cv--------------------------")
        predicted = cross_val_predict(clf, X, y, cv=cv, n_jobs=cpu_num)
        # y_predict_prob = cross_val_predict(clf, X, y, cv=cv, n_jobs=cpu_num, method='predict_proba')
        # ROC_AUC_area = metrics.roc_auc_score(y, y_predict_prob[:, 1])
        # print("AUC:{}".format(ROC_AUC_area))
        print("ACC:{}".format(metrics.accuracy_score(y, predicted)))
        print("MCC:{}\n".format(metrics.matthews_corrcoef(y, predicted)))
        print(classification_report(y, predicted))
        print("confusion matrix\n")
        print(pd.crosstab(pd.Series(y, name='Actual'), pd.Series(predicted, name='Predicted')))
    else:
        clf.fit(X, y)
        joblib.dump(clf, model)


# main
def main():
    parser = argparse.ArgumentParser()
    # train set
    parser.add_argument('-i', required=True, nargs="+", help="trainfasta file name")
    parser.add_argument('-o', required=True, help="model")
    # word2vec
    parser.add_argument('-b', default=0, help="Fill in the vector")
    parser.add_argument('-sg', type=int, default=1, help="")
    parser.add_argument('-iter', type=int, default=5, help="")
    parser.add_argument('-hs', type=int, default=0, help="")
    parser.add_argument('-premodel', help="pretrainmodel")
    parser.add_argument('-window_size', type=int, default=20, help="window size")
    parser.add_argument('-model_name', default="model.model", help="embedding model")
    parser.add_argument('-hidden_size', type=int, default=100, help="The dimension of word")
    # svm
    parser.add_argument('-mms', type=bool, default=False, help="minmaxscaler")
    parser.add_argument('-ss', type=bool, default=False, help="StandardScaler")
    parser.add_argument('-cv', type=int, help="cross validation")
    parser.add_argument('-n_job', '-n', default=-1, help="num of thread")
    parser.add_argument('-grad', type=bool, default=False, help="grad")
    # splite
    parser.add_argument('-kmer', '-k', type=int, default=3, help="k-mer: k size")
    parser.add_argument('-splite', '-s', type=int, default=0, help="kmer splite(0) or normal splite(1)")

    args = parser.parse_args()
    print(args)

    if args.splite == 0:
        print("kmer splite !")
    else:
        print("normal splite !")

    y = []
    for i in args.i:
        f = open(i).readlines()
        y.append(int(len(f) / 2))

    print(y)
    num_y = len(args.i)
    train_y = []
    for i in range(num_y):
        train_y += [i] * y[i]

    start_time = time.time()

    train_word = splite_word(args.i, args.kmer, args.splite)

    csv = tocsv(train_word, args.sg, args.hs, args.window_size, args.hidden_size, args.model_name,
                args.b, args.iter, args.premodel)

    svm(csv, train_y, args.cv, args.n_job, args.mms, args.ss, args.grad, args.o)

    end_time = time.time()
    print("end ............................")
    print("Time consuming：{}s\n".format(end_time - start_time))


if __name__ == '__main__':
    main()
