import itertools

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from DCiPatho_config import Config

config = Config()

complements = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
nt_bits = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def get_rc(seq):
    ''' Return the reverse complement of seq
    '''
    rev = reversed(seq)
    return "".join([complements.get(i, i) for i in rev])


def mer2bits(kmer):
    ''' convert kmer to bit representation
    '''
    bit_mer = nt_bits[kmer[0]]
    for c in kmer[1:]:
        bit_mer = (bit_mer << 2) | nt_bits[c]
    return bit_mer


def compute_kmer(ks):
    ''' Get the indeces of each canonical kmer in the kmer count vectors
     获取kmer计数向量中每个规范kmer的索引
    '''
    kmer_list = []
    kmer_inds = {k: {} for k in ks}
    kmer_count_lens = {k: 0 for k in ks}

    alphabet = 'ACGT'
    for k in ks:
        all_kmers = [''.join(kmer) for kmer in itertools.product(alphabet, repeat=k)]
        # 生成所有kmers的组合 4的k次方个
        all_kmers.sort()
        # 排序 以ascii码大小排
        ind = 0
        for kmer in all_kmers:
            bit_mer = mer2bits(kmer)
            rc_bit_mer = mer2bits(get_rc(kmer))
            if rc_bit_mer in kmer_inds[k]:
                kmer_inds[k][bit_mer] = kmer_inds[k][rc_bit_mer]
            else:
                kmer_list.append(kmer)
                kmer_inds[k][bit_mer] = ind
                kmer_count_lens[k] += 1
                ind += 1
    return kmer_inds, kmer_count_lens, kmer_list


def getTrainData(df):
    print(df.columns)
    dense_features_col = [col for col in df.columns]
    return dense_features_col

def data_reprocess_from_csv():
    patho = np.load(config.patho_path)
    print('pathogen numbers:', patho.shape[0])
    nonpatho = np.load(config.nonpatho_path)
    print('nonpathogen numbers:', nonpatho.shape[0])

    train_df = np.concatenate((patho, nonpatho), axis=0)
    labels = np.concatenate((np.ones(patho.shape[0]), np.zeros(nonpatho.shape[0])))

    print(len(labels))
    scaler = StandardScaler().fit(train_df)
    X = scaler.transform(train_df)

    # X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1,
    #                                                     random_state=config.random_state)
    print('X_train:', len(X))
    # print('X_test nums:',len(X_test))
    return X, labels

def data_reprocess_from_np():
    patho = np.load(config.patho_path)
    print('pathogen numbers:', patho.shape[0])
    nonpatho = np.load(config.nonpatho_path)
    print('nonpathogen numbers:', nonpatho.shape[0])

    train_df = np.concatenate((patho, nonpatho), axis=0)
    labels = np.concatenate((np.ones(patho.shape[0]), np.zeros(nonpatho.shape[0])))

    print(len(labels))
    scaler = StandardScaler().fit(train_df)
    X = scaler.transform(train_df)

    # X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1,
    #                                                     random_state=config.random_state)
    print('X_train:', len(X))
    # print('X_test nums:',len(X_test))
    return X, labels


def data_preprocess_for_predict(path):
    patho = np.load(path)
    print('nums of combined fasta:',patho.shape)
    scaler = StandardScaler().fit(patho)
    X = scaler.transform(patho)
    return X


def DataPreprocess():
    # np way:
    patho = np.load(config.patho_path)
    print('pathogen numbers:', patho.shape[0])
    nonpatho = np.load(config.nonpatho_path)
    print('nonpathogen numbers:', nonpatho.shape[0])

    train_df = np.concatenate((patho, nonpatho), axis=0)
    labels = np.concatenate((np.ones(patho.shape[0]), np.zeros(nonpatho.shape[0])))

    print(len(labels))
    scaler = StandardScaler().fit(train_df)
    X = scaler.transform(train_df)

    X_train, X_left, y_train, y_left = train_test_split(X, labels, test_size=config.val_size,
                                                        random_state=config.random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_left, y_left, test_size=config.test_size,
                                                    random_state=config.random_state)
    print('train nums:', len(X_train))
    print('validation nums:', len(X_val))
    print('test nums:', len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test


def allFreqsPreprocess():
    data = pd.read_csv(config.all_freqs, header=None, nrows=30000)
    names = []
    for i in range(data.shape[1]):
        names.append('F' + str(i))
    data.columns = names
    scaler = StandardScaler().fit(data)
    X = scaler.transform(data)
    label = pd.read_csv(config.all_labels)
    label = label['label'].tolist()
    print(label)
    print(len(label))
    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=config.test_size,
                                                        random_state=config.random_state)
    print('train nums:', len(y_train))
    print('train nums:', len(X_train))
    print('test nums:', len(X_test))
    return data, X_train, X_test, y_train, y_test


def testDataPreprocess():
    # 878

    patho = pd.read_csv(config.test_878_patho_path, header=None, skiprows=1)
    # patho = pd.read_csv(config.test_patho_path, header=None)
    # nonpatho = pd.read_csv(config.test_nonpatho_path, header=None)

    # train_df = pd.concat((patho, nonpatho))

    print('read csv over')

    # train_df = pd.concat((patho, nonpatho))
    print(patho.shape)
    # print(nonpatho.shape)
    # labels = np.concatenate((np.zeros(patho.shape[0]), np.ones(nonpatho.shape[0])))
    labels = (np.ones(patho.shape[0]))

    print(len(labels))
    scaler = StandardScaler().fit(patho)
    X = scaler.transform(patho)
    # scaler = StandardScaler().fit(train_df)
    # X = scaler.transform(train_df)

    return X, labels


def paint(train_loss, test_acc, test_roc, test_f1):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    # plt.plot(test_loss, label='Validation Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_roc, label='Roc-auc score', color='black', linestyle='dotted')
    plt.plot(test_f1, label='F1 score')
    plt.plot(test_acc, label='Accuracy')

    # plt.plot(test_loss, label='Validation Loss')
    plt.title('Validation eval')
    plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    plt.legend()
    plt.show()


def roc(y_ture, y_pred_probs):
    y_test = y_ture.cpu()
    y_pred_probs = y_pred_probs.cpu().detach().numpy()
    roc_score = roc_auc_score(y_test, y_pred_probs)
    print('AUC:', roc_score)
    return roc_score


def F1(y_ture, y_pred):
    y_test = y_ture.cpu()
    y_pred = y_pred.cpu()
    f1 = f1_score(y_test, y_pred)
    print('F1 Score: %f' % f1)

    return f1


def mcc(y_ture, y_pred):
    y_test = y_ture.cpu()
    y_pred = y_pred.cpu()
    mcc = matthews_corrcoef(y_test, y_pred)
    print('MCC: %f' % mcc)
    return mcc


def acc(y_ture, y_pred):
    y_test = y_ture.cpu()
    y_pred = y_pred.cpu()

    acc = accuracy_score(y_test, y_pred)
    print('ACC: %f' % acc)
    return acc
