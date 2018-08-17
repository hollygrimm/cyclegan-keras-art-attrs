
class BaseDataLoader(object):
    def __init__(self, config):
        self.config = config

    def get_trainA_data(self):
        raise NotImplementedError

    def get_trainB_data(self):
        raise NotImplementedError

    def get_testB_data(self):
        raise NotImplementedError      

    def get_testA_data(self):
        raise NotImplementedError      