import time


class Config:
    def __init__(self):
        self.patho_path = r'example_data/patho_141.csv'
        self.test_patho_path = ''
        self.nonpatho_path = r'example_data/nonpatho_517.csv'
        self.test_nonpatho_path = ''
        # for one freqs file
        self.all_freqs = ''
        self.all_labels = ''
        self.test_path = ''
        # set ResNet module
        self.hidden_layers = [1024, 512, 256]
        # set DeepNet module
        self.deep_layers = [4096, 2048, 1024, 512]
        # set CrossNet layers
        self.num_cross_layers = 7
        self.end_dims = [2048, 1024, 512, 256]
        self.out_layer_dims = 1024
        self.test_size = 0.2
        self.random_state = 1
        self.num_epoch = 350
        self.patience = 30
        self.batch_size = 128
        self.Dropout = 0.1
        self.lr = 0.000001
        self.l2_regularization = 0.000001
        self.device_id = 0
        self.use_cuda = True
        self.save_model = False
        self.model_name = 'dci.model'


