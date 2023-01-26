class Config:
    def __init__(self):
        # self.patho_path = r'example_data/patho_141.csv'
        # self.patho_path = 'example_data/patho_freq1500.csv'
        self.patho_path = r'C:\Users\lie\Desktop\cal_kmer\combine_part_npy\1228patho.npy'
        # self.patho_path = r'E:\combine\dataset\k=7_plas_freqall.csv'
        # 878 data
        self.test_878_patho_path = r'E:\combine\dataset\878patho_test.csv'
        #
        self.test_patho_path = r'D:\ResDeepCross\Cal_kmer\data\patho_141.csv'
        self.test_nonpatho_path = r'D:\ResDeepCross\Cal_kmer\data\nonpatho_517.csv'

        # self.nonpatho_path = r'example_data/nonpatho_517.csv'
        # self.nonpatho_path = 'example_data/nonpatho_freq1500.csv'
        self.nonpatho_path = r'C:\Users\lie\Desktop\cal_kmer\combine_part_npy\1228nonpatho.npy'

        self.nums_dic = {3: 32, 4: 136, 5: 512, 6: 2080, 7: 8192, 'all': 10952}
        self.freqs_nums = self.nums_dic['all']
        self.k5_idx = [168, 680]
        self.k6_idx = [680, 2760]
        self.k7_idx = [2760, 10952]
        # for one freqs file
        self.all_freqs = ''
        self.all_labels = ''
        self.test_path = ''
        # set ResNet module
        # self.hidden_layers = [1024, 512, 256]
        self.hidden_layers = [512, 256, 128]
        # set DeepNet module
        self.deep_layers = [4096, 2048, 1024, 512]
        # set CrossNet layers
        self.num_cross_layers = 7
        # self.k_fold = 10
        # self.end_dims = [2048, 1024, 512, 256]
        self.end_dims = [1024, 512, 256]
        self.out_layer_dims = 1024
        self.val_size = 0.2
        # remove half val data to test data
        self.test_size = 0.5
        self.random_state = 1
        self.num_epoch = 220
        self.patience = 30
        self.batch_size = 512
        self.Dropout = 0.3
        self.lr = 0.0000001
        self.l2_regularization = 0.00001
        self.device_id = 0
        self.use_cuda = True
        # self.save_model = False
        self.save_model = True
        # self.model_name = 'models/k3-7_model.pt'
        self.output_base_path = 'C:/Users/lie/Desktop/BIOS/DCiPatho/models/'
        self.best_model_name = '_best_k3-7_model.pt'
        # self.save_name = 'models/cross_model.pt'
