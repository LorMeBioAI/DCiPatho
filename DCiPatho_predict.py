import time
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from DCiPatho_network import DCiPatho
from DCiPatho_config import Config
from Utils.tool import testDataPreprocess, getTrainData, DataPreprocess


# use trained model to predict 

def predict(model, test_data, y_test, save_file=None):
    model.cuda()
    model.load_state_dict(torch.load(config.model_name))
    y_pred_probs = model(test_data)
    y_pred = torch.where(y_pred_probs > 0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
    accuracy = accuracy_score(y_test.cpu(), y_pred.cpu())
    # roc = roc_auc_score(y_test.cpu(), y_pred.cpu())
    # f1 = f1_score(y_test.cpu(), y_pred.cpu())
    # mcc = matthews_corrcoef(y_test.cpu(), y_pred.cpu())
    print('accuracy:', accuracy)
    # print('f1:', f1)
    # print('roc:', roc)
    # print('mcc:', mcc)
    if save_file:
        y_pred_probs = y_pred_probs.cpu().tolist()
        y_pred = y_pred.cpu().tolist()
        d = {'y_pred_probs': y_pred_probs, 'y_pred': y_pred}
        df = pd.DataFrame(d)
        df.to_csv(save_file + '.csv')


if __name__ == '__main__':
    config = Config()
    s = time.time()
    df_X, X, labels = testDataPreprocess()
    # df_X, X, labels = testDataPreprocess()
    dense_features_cols = getTrainData(df_X)
    model = DCiPatho(dense_features_cols)
    test_data, y_test = torch.tensor(X).float().cuda(), torch.tensor(labels).float().cuda()
    predict(model, test_data, y_test)
    print('costs:', time.time() - s)
