import pandas as pd
import torch
import torch.utils.data as Data
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from DCiPatho_config import Config
from DCiPatho_network import DCiPatho
from DCiPatho_trainer import Trainer
from Utils.tool import roc, acc, F1, mcc, data_reprocess_from_csv

config = Config()


def save_res(test_acc, test_roc, test_f1, test_mcc):
    print('ACC:')
    print(pd.DataFrame(test_acc).describe())
    print('ROC:')
    print(pd.DataFrame(test_roc).describe())
    print('F1 SCORE:')
    print(pd.DataFrame(test_f1).describe())
    print('MCC:')
    print(pd.DataFrame(test_mcc).describe())


# k-fold method
def train_and_eval_with_cv():
    X, y = data_reprocess_from_csv()
    kf = KFold(n_splits=config.fold, shuffle=True)
    val_roc, val_acc, val_f1, val_mcc = [], [], [], []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        print(f'fold_{fold} train nums:', len(X_train))
        print(f'fold_{fold}X_val nums:', len(X_val))
        print(f'fold_{fold}y nums:', len(y_train))
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
        val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float())
        model = DCiPatho()
        model.cuda()

        trainer = Trainer(model=model)
        trainer.use_cuda()

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False)
        # Set initial acc to reduce calculation
        best_acc = 0.90
        train_loss = []
        val_data, y_val = torch.tensor(X_val).cuda(), torch.tensor(y_val).cuda()

        for epoch in range(config.num_epoch):
            print('-' * 20 + f' Fold {fold} Epoch {epoch + 1} starts ' + '-' * 20)
            trainer._train_an_epoch(train_loader, epoch, train_loss)
            if epoch >= 50:
                model.eval()
                torch.cuda.empty_cache()
                with torch.no_grad():
                    y_pred_probs = model(val_data.float())
                    y_pred = torch.where(y_pred_probs > 0.5, torch.ones_like(y_pred_probs),
                                         torch.zeros_like(y_pred_probs))
                    this_roc = roc(y_val, y_pred_probs)
                    val_roc.append(this_roc)
                    this_acc = acc(y_val, y_pred)
                    val_acc.append(this_acc)
                    this_f1 = F1(y_val, y_pred)
                    val_f1.append(this_f1)
                    this_mcc = mcc(y_val, y_pred)
                    val_mcc.append(this_mcc)
                    if this_acc > best_acc:
                        print(f'fold_{fold} best metrics on validation set: ')
                        print(f'acc:{this_acc}')
                        print(f'auc:{this_roc}')
                        print(f'f1:{this_f1} ')
                        print(f'mcc:{this_mcc} ')
                        best_acc = this_acc
                        this_fold_best_model = f"{config.output_base_path}fold_{fold}_{config.best_model_name}"
                        model.saveModel(this_fold_best_model)
    # print(this_fold_best_model)
    # test_best_model(X_test, y_test, this_fold_best_model)
    # print(f"Validation Results:")
    # print(f"ROC: {sum(val_roc) / n_splits:.4f}")
    # print(f"ACC: {sum(val_acc) / n_splits:.4f}")
    # print(f"F1: {sum(val_f1) / n_splits:.4f}")
    # print(f"MCC: {sum(val_mcc) / n_splits:.4f}")


# Training and validation process of the non-k-fold method
def train_and_eval(X_train, X_val, y_train, y_val):
    print('use cuda?', torch.cuda.is_available())
    best_acc = 0.90
    # early_stopping = EarlyStopping(patience=patience, verbose=True)
    val_acc = [0]
    val_roc = []
    val_f1 = []
    val_mcc = []
    train_dataset = Data.TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
    model = DCiPatho()
    # model = DCiPatho2()
    model = model.cuda()
    print(model)

    ####################################################################################
    # train model
    ####################################################################################

    trainer = Trainer(model=model)
    trainer.use_cuda()
    val_data, y_val = torch.tensor(X_val).cuda(), torch.tensor(y_val).cuda()
    data_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)

    train_loss = []
    for epoch in range(config.num_epoch):
        print('-' * 20 + ' Epoch {} starts '.format(epoch) + '-' * 20)
        trainer._train_an_epoch(data_loader, epoch, train_loss)

        ####################################################################################
        # test
        ###################################################################################
        if epoch >= 120:
            model.eval()
            torch.cuda.empty_cache()
            # y_pred_probs = (torch.tensor(test_data).float())
            print('model on validation set...')

            with torch.no_grad():
                y_pred_probs = model(val_data.float())
                y_pred = torch.where(y_pred_probs > 0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
                val_roc.append(roc(y_val, y_pred_probs))
                this_acc = acc(y_val, y_pred)
                val_acc.append(this_acc)
                val_f1.append(F1(y_val, y_pred))
                val_mcc.append(mcc(y_val, y_pred))
                if this_acc > best_acc:
                    print('best ACC on validation set:', this_acc)
                    best_acc = this_acc
                    # model.save(model_name)
                    model.saveModel(config.output_base_path + str(this_acc)[:5] + config.best_model_name)

    # save
    # if config.save_model:
    #     trainer.save()
    # show
    # save_res(test_acc, test_roc, test_f1, test_mcc)


def test_best_model(x_test, y_test, model_path):
    model = DCiPatho()
    model.cuda()
    print('loading best model:', model_path)
    model.load_state_dict(torch.load(model_path))
    test_data, y_test = torch.tensor(x_test).float().cuda(), torch.tensor(y_test).float().cuda()
    model.eval()
    with torch.no_grad():
        y_pred_probs = model(test_data)
        y_pred = torch.where(y_pred_probs > 0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
        accuracy = acc(y_test.cpu(), y_pred.cpu())
        ROC = roc(y_test.cpu(), y_pred.cpu())
        f1 = F1(y_test.cpu(), y_pred.cpu())
        MCC = mcc(y_test.cpu(), y_pred.cpu())
        print('test acc:', accuracy)
        print('test f1:', f1)
        print('test roc:', ROC)
        print('test mcc:', MCC)


if __name__ == "__main__":
    train_and_eval_with_cv()
    # X, y = data_reprocess_return_xy()
    # kf = KFold(n_splits=5, shuffle=True)
    # val_roc, val_acc, val_f1, val_mcc = [], [], [], []
    # for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    #     X_train, X_val = X[train_idx], X[val_idx]
    #     y_train, y_val = y[train_idx], y[val_idx]
    #     model_path = r'E:\models\Dcipatho_pretrain_model\fold_1_cv5_best_k3-7_model.pt'
    #     test_best_model(X_val, y_val, model_path)
    #     break
    # this_time_model_path = config.best_model_name
    # X_train, X_val, X_test, y_train, y_val, y_test = DataPreprocess()
    # train_and_eval(X_train, X_val, y_train, y_val)
    # test_best_model(X_test, y_test)
