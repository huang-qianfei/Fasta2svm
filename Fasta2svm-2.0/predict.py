import sys

sys.path.extend(["../../", "../", "./"])
import warnings

warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
import gensim
from sklearn.metrics import classification_report
from sklearn import metrics
import time
import argparse
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
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


def tocsv(train_word, b, premodel):

    model = gensim.models.KeyedVectors.load_word2vec_format(premodel, binary=False)

    csv = save_csv(train_word, model, b)

    return csv


# svm
def svm(testcsv, test_y, n_job, mms, ss, model):

    X1 = testcsv
    y1 = test_y
    clf = joblib.load(model)

    if mms:
        print("MinMaxScaler")
        minMax = MinMaxScaler()
        minMax.fit(X1)
        X1 = minMax.transform(X1)

    if ss:
        print("StandardScaler")
        scaler = StandardScaler()
        scaler.fit(X1)
        X1 = scaler.transform(X1)

    pre = clf.predict(X1)
    print("ACC:{}".format(metrics.accuracy_score(y1, pre)))
    print("MCC:{}".format(metrics.matthews_corrcoef(y1, pre)))
    print(classification_report(y1, pre))
    print("confusion matrix\n")
    print(pd.crosstab(pd.Series(y1, name='Actual'), pd.Series(pre, name='Predicted')))


# main
def main():
    parser = argparse.ArgumentParser()
    # train set
    parser.add_argument('-t', required=True, nargs="+", help="testfasta file name")
    parser.add_argument('-m', required=True, help="model")
    parser.add_argument('-em', required=True, default="model.model", help="embedding model")
    parser.add_argument('-b', default=0, help="Fill in the vector")
    # svm
    parser.add_argument('-mms', type=bool, default=False, help="minmaxscaler")
    parser.add_argument('-ss', type=bool, default=False, help="StandardScaler")
    parser.add_argument('-n_job', '-n', default=-1, help="num of thread")
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
    for i in args.t:
        f = open(i).readlines()
        y.append(int(len(f) / 2))

    print(y)
    num_y = len(args.t)
    test_y = []
    for i in range(num_y):
        test_y += [i] * y[i]

    start_time = time.time()

    train_word = splite_word(args.t, args.kmer, args.splite)

    csv = tocsv(train_word, args.b, args.em)

    svm(csv, test_y, args.n_job, args.mms, args.ss, args.m)

    end_time = time.time()
    print("end ............................")
    print("Time consuming：{}s\n".format(end_time - start_time))


if __name__ == '__main__':
    main()
