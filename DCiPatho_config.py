class Config:
    def __init__(self):
        #################################################################
        #  settings for train and eval
        #################################################################
        # set the npy or csv data path here
        self.patho_path = ''
        self.nonpatho_path = ''
        self.nums_dic = {3: 32, 4: 136, 5: 512, 6: 2080, 7: 8192, 'all': 10952}
        self.freqs_nums = self.nums_dic['all']
        # set ResNet module
        self.hidden_layers = [512, 256, 128]
        # set DeepNet module
        self.deep_layers = [4096, 2048, 1024, 512]
        # set CrossNet layers
        self.num_cross_layers = 7
        self.end_dims = [1024, 512, 256]
        self.out_layer_dims = 1024
        self.val_size = 0.2
        self.fold = 5
        # remove half val data to test data
        self.test_size = 0.5
        self.random_state = 1
        self.num_epoch = 200
        self.patience = 30
        self.batch_size = 512
        self.Dropout = 0.3
        self.lr = 0.0000002
        self.l2_regularization = 0.00001
        self.device_id = 0
        self.use_cuda = True
        self.save_model = True
        self.output_base_path = ''
        self.best_model_name = 'fold_1_cv5_best_k3-7_model.pt'

        ########################################################################
        #  settings for predict raw fasta files
        ########################################################################

        # set the following parameters to the directory of the folder you need to predict
        self.raw_fasta_path = 'example_data'
        # set combined_fasta_path if needed or just by default
        self.combined_fasta_path = 'temp_data/temp_combined_fna.fasta'
        #  settings for calculation of kmer frequency of fasta
        self.ks = [3, 4, 5, 6, 7]
        self.num_procs = 8
        self.freqs_file = 'temp_data/freqs_file.npy'
        self.save_res_path = 'temp_data/results_for_pre.csv'
