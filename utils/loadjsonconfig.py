import json

class LoadJsonConfig:
    ''' 
    json_file: path to json config file 
    '''
    def __init__(self, json_file):
        self.json_file = json_file
        self.config = self.__read_json()
        self.NO_EPOCH = self.config['epochs']
        self.BATCH_SIZE = self.config['batch_size']
        self.GPU = self.config['gpu']
        self.LOGS_DIR = self.config['logs_dir']
        self.DATASET_PATH = self.config['dataset_path']
        self.DROP_OUT = self.config['drop_out']
        self.OPTIMIZER = self.config['optimizer']
        self.MOMENTUM = self.config['momentum']
        self.LEARNING_RATE = self.config['leaning_rate']
        self.WEIGHT_DECAY = self.config['weight_decay']
        self.NUM_WORKERS = self.config['num_workers']
        self.CLASS_NAME = [class_name for class_name in self.config['classes']]
        self.USE_CLASS_WEIGHT = self.config['use_class_weight']
        self.PASS_THRESHOLD = self.config['pass_threshold']
        self.FAIL_THRESHOLD = self.config['fail_threshold']
        self.RESIZE = self.config['resize']
        self.PADDING_SIZE = self.config['padding_size']
        self.MEAN = self.config['mean']
        self.STD = self.config['std']


    def __read_json(self):
        with open(self.json_file) as f:
            config = json.load(f)
        return config

