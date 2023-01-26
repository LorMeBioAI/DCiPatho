import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from DCiPatho_config import Config

config = Config()


class Trainer(object):
    def __init__(self, model):
        self._model = model
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=config.lr,
                                           weight_decay=config.l2_regularization)
        self._loss_func = torch.nn.BCELoss()

    def _train_single_batch(self, x, labels):
        self._optimizer.zero_grad()
        y_predict = self._model(x)
        #
        # y_predict = torch.Tensor(y_predict)
        loss = self._loss_func(y_predict.view(-1), labels)
        # loss = self._loss_func(y_predict[0].view(-1), labels)

        loss.backward()
        self._optimizer.step()

        loss = loss.item()
        return loss, y_predict

    def _train_an_epoch(self, train_loader, epoch_id, train_loss):
        self._model.train()
        total = 0
        for batch_id, (x, labels) in enumerate(train_loader):
            x = Variable(x)
            labels = Variable(labels)
            if config.use_cuda is True:
                x, labels = x.cuda(), labels.cuda()

            loss, predicted = self._train_single_batch(x, labels)

            total += loss
            # print('[Training Epoch: {}] Batch: {}, Loss: {}'.format(epoch_id, batch_id, loss))
        print("Training Epoch: %d, total loss: %f" % (epoch_id, total))
        train_loss.append(total)

    def train(self, train_dataset):
        self.use_cuda()
        for epoch in range(config.num_epoch):
            print('-' * 20 + ' Epoch {} starts '.format(epoch) + '-' * 20)
            data_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
            self._train_an_epoch(data_loader, epoch_id=epoch)

    def use_cuda(self):
        if config.use_cuda is True:
            assert torch.cuda.is_available(), 'CUDA is not available'
            torch.cuda.set_device(config.device_id)
            self._model.cuda()

    def save(self,model_name):
        self._model.saveModel(model_name)
