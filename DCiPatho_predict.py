import time

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef

from DCiPatho_config import Config
from DCiPatho_network import DCiPatho
from Utils import cal
from Utils.combine_fna import combine
from Utils.tool import data_preprocess_for_predict


def load_model(path):
    model = DCiPatho().eval()
    if config.use_cuda:
        model.load_state_dict(torch.load(path))
        print('model loaded using cuda')
    else:
        model.load_state_dict(torch.load(path), map_location=torch.device('cpu'))
        print('model loaded using cpu')
    return model


# use trained model to predict
def predict(y_test=False):
    model = load_model(config.best_model_name)
    combine(config.raw_fasta_path, config.combined_fasta_path)
    print('calculate kmer freqs..')
    names = cal.cal_main(config.combined_fasta_path, config.num_procs, config.ks, config.freqs_file)
    # sentences_idx = load_dataset_word2vec(message)
    X = data_preprocess_for_predict(config.freqs_file)
    X = torch.tensor(X).float()
    y_pred_probs = model(X)
    # y_pred = torch.where(y_pred_probs > 0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
    # y_pred = torch.where(y_pred_probs > 0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
    y_pred = y_pred_probs.detach().cpu().numpy().tolist()
    # y_pred = y_pred.cpu().numpy().tolist()
    print('my_prediction', y_pred)
    res = []
    for i, d in enumerate(names):
        res.append({'name': names[i], 'value': y_pred[i]})
    print(res)

    y_pred = torch.where(y_pred_probs > 0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
    # save if needed
    if config.save_res_path:
        y_pred_probs = y_pred_probs.cpu().tolist()
        y_pred = y_pred.cpu().tolist()
        d = {'names': names, 'y_pred_probs': y_pred_probs, 'y_pred': y_pred}
        df = pd.DataFrame(d)
        df.to_csv(config.save_res_path)
    # If there are labels for test_data, evaluate here
    if y_test:
        accuracy = accuracy_score(y_test.cpu(), y_pred.cpu())
        roc = roc_auc_score(y_test.cpu(), y_pred.cpu())
        f1 = f1_score(y_test.cpu(), y_pred.cpu())
        mcc = matthews_corrcoef(y_test.cpu(), y_pred.cpu())
        print('accuracy:', accuracy)
        print('f1:', f1)
        print('roc:', roc)
        print('mcc:', mcc)


if __name__ == '__main__':
    config = Config()
    s = time.time()
    predict()
    print('costs:', time.time() - s)
