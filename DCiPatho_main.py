import pandas as pd
import torch
# from DCiPatho_network2 import DCiPatho2
import torch.utils.data as Data
from torch.utils.data import DataLoader

from DCiPatho_config import Config
from DCiPatho_network import DCiPatho
from DCiPatho_trainer import Trainer
from Utils.tool import DataPreprocess, roc, acc, F1, mcc

config = Config()


def save_res(test_acc, test_roc, test_f1, test_mcc):
    # pd.DataFrame(test_acc)

    print('ACC:')
    print(pd.DataFrame(test_acc).describe())
    print('ROC:')
    print(pd.DataFrame(test_roc).describe())
    print('F1 SCORE:')
    print(pd.DataFrame(test_f1).describe())
    print('MCC:')
    print(pd.DataFrame(test_mcc).describe())


def train_and_eval(X_train, X_val, y_train, y_val):
    print('use cuda?', torch.cuda.is_available())
    best_acc = 0.95
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
        if epoch >= 100:
            model.eval()
            torch.cuda.empty_cache()
            # y_pred_probs = (torch.tensor(test_data).float())
            print('model on validation set...')

            with torch.no_grad():
                y_pred_probs = model(val_data.float())
                #

                # y_pred_probs = y_pred_probs[0]
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
    # this_time_model_path = config.best_model_name
    X_train, X_val, X_test, y_train, y_val, y_test = DataPreprocess()
    train_and_eval(X_train, X_val, y_train, y_val)
    # test_best_model(X_test, y_test)
