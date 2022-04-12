import torch
from torch.utils.data import DataLoader
from Utils.tool import mcc, roc, F1, acc, DataPreprocess, getTrainData, testDataPreprocess
from DCiPatho_trainer import Trainer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from DCiPatho_network import DCiPatho
import torch.utils.data as Data
import pandas as pd
from DCiPatho_config import Config

config = Config()


def train_and_eval(train_df, X_train, X_test, y_train, y_test):
    # early_stopping = EarlyStopping(patience=patience, verbose=True)
    test_acc = []
    test_roc = []
    test_f1 = []
    test_mcc = []
    dense_features_cols = getTrainData(train_df)
    train_dataset = Data.TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
    model = DCiPatho(dense_features_cols)
    model = model.cuda()
    print(model)

    ####################################################################################
    # train model
    ####################################################################################

    trainer = Trainer(model=model)
    trainer.use_cuda()
    test_data, y_test = torch.tensor(X_test).cuda(), torch.tensor(y_test).cuda()
    train_loss = []
    for epoch in range(config.num_epoch):
        print('-' * 20 + ' Epoch {} starts '.format(epoch) + '-' * 20)
        data_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
        trainer._train_an_epoch(data_loader, epoch, train_loss)

        ####################################################################################
        # test
        ###################################################################################
        model.eval()
        # y_pred_probs = (torch.tensor(test_data).float())
        y_pred_probs = model(test_data.float())
        y_pred = torch.where(y_pred_probs > 0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))

        test_roc.append(roc(y_test, y_pred_probs))
        test_acc.append(acc(y_test, y_pred))
        test_f1.append(F1(y_test, y_pred))
        test_mcc.append(mcc(y_test, y_pred))

    # save
    if config.save_model:
        trainer.save()
    # show
    print('ACC:')
    print(pd.DataFrame(test_acc).describe())
    print('ROC:')
    print(pd.DataFrame(test_roc).describe())
    print('F1 SCORE:')
    print(pd.DataFrame(test_f1).describe())
    print('MCC:')
    print(pd.DataFrame(test_mcc).describe())


if __name__ == "__main__":
    train_df, X_train, X_test, y_train, y_test = DataPreprocess()
    train_and_eval(train_df, X_train, X_test, y_train, y_test)
