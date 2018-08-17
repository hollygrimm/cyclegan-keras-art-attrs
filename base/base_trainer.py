
class BaseTrain(object):
    def __init__(self, model, trainA_data, trainB_data, testA_data, testB_data, config):
        self.model = model
        self.trainA_data = trainA_data
        self.trainB_data = trainB_data
        self.testA_data = testA_data
        self.testB_data = testB_data
        self.config = config

    def train(self):
        raise NotImplementedError