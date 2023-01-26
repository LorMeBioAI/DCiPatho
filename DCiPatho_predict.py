import time

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef

from DCiPatho_config import Config
from DCiPatho_network import DCiPatho
from Utils.tool import DataPreprocess


# use trained model to predict 

def predict(model, test_data, y_test=False, eval=False, save_file=None):
    y_pred_probs = model(test_data)
    y_pred = torch.where(y_pred_probs > 0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
    if eval:
        accuracy = accuracy_score(y_test.cpu(), y_pred.cpu())
        roc = roc_auc_score(y_test.cpu(), y_pred.cpu())
        f1 = f1_score(y_test.cpu(), y_pred.cpu())
        mcc = matthews_corrcoef(y_test.cpu(), y_pred.cpu())
        print('accuracy:', accuracy)
        print('f1:', f1)
        print('roc:', roc)
        print('mcc:', mcc)
        if len(save_file) > 0:
            y_pred_probs = y_pred_probs.cpu().tolist()
            y_pred = y_pred.cpu().tolist()
            d = {'y_pred_probs': y_pred_probs, 'y_pred': y_pred}
            df = pd.DataFrame(d)
            df.to_csv(save_file + '.csv')


if __name__ == '__main__':
    config = Config()
    s = time.time()
    # X_test, y_test = testDataPreprocess()
    X_train, X_val, X_test, y_train, y_val, y_test = DataPreprocess()

    # df_X, X, labels = testDataPreprocess()
    # dense_features_cols = getTrainData(df_X)
    model = DCiPatho()
    model.cuda()
    model.load_state_dict(torch.load('models/0.950_best_k3-7_model.pt'))
    test_data, y_test = torch.tensor(X_test).float().cuda(), torch.tensor(y_test).float().cuda()
    output_path = 'output/1.13predict_x_test.csv'
    predict(model, test_data, y_test, eval=True, save_file=output_path)
    print('costs:', time.time() - s)
