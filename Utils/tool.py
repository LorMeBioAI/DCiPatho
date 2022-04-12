from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve, \
    matthews_corrcoef
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from DCiPatho_config import Config

config = Config()


def getTrainData(df):
    print(df.columns)
    dense_features_col = [col for col in df.columns]
    return dense_features_col


def DataPreprocess():
    # patho = pd.read_csv(config.test_patho_path, header=None)
    patho = pd.read_csv(config.patho_path, header=None)
    names = []
    for i in range(patho.shape[1]):
        names.append('F' + str(i))
    patho.columns = names
    nonpatho = pd.read_csv(config.nonpatho_path, header=None)
    nonpatho.columns = names
    print('read csv over')

    train_df = pd.concat((patho, nonpatho))
    print(train_df.shape)
    labels = np.concatenate((np.ones(patho.shape[0]), np.zeros(nonpatho.shape[0])))

    print(len(labels))
    scaler = StandardScaler().fit(train_df)
    X = scaler.transform(train_df)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=config.test_size,
                                                        random_state=config.random_state)
    print('train nums:', len(X_train))
    print('test nums:', len(X_test))
    return train_df, X_train, X_test, y_train, y_test


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
    test_path = config.test_path
    # patho = pd.read_csv(test_path, header=None, skiprows=1)
    patho = pd.read_csv(test_path, header=None)
    names = []
    for i in range(patho.shape[1]):
        names.append('F' + str(i))
    patho.columns = names
    # nonpatho = pd.read_csv(config.nonpatho_path)
    # nonpatho.columns = names
    print('read csv over')

    # train_df = pd.concat((patho, nonpatho))
    print(patho.shape)
    # labels = np.concatenate((np.zeros(patho.shape[0]), np.ones(nonpatho.shape[0])))
    labels = (np.ones(patho.shape[0]))

    print(len(labels))
    scaler = StandardScaler().fit(patho)
    scaled = scaler.transform(patho)
    X = scaled
    train_df = patho
    return train_df, X, labels


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
    print('ROC:', roc_score)
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
