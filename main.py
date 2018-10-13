import sys
from utils.utils import get_args, process_config, create_dirs
from data_loader.data_loader import DataLoader
from models.cyclegan_attr_model import CycleGANAttrModel
from trainers.cyclegan_trainer import CycleGANModelTrainer

# TODO: Add proper logging

def main():
    # get json configuration filepath from the run argument
    # process the json configuration file
    args = get_args()
    config = process_config(args.config)

    # create the experiment directories
    log_dir, checkpoint_dir = create_dirs(config)

    print('Create the data generator')
    data_loader = DataLoader(config)

    print('Create the model')
    model = CycleGANAttrModel(config)
    model.build_model()
    print('model ready loading data now')

    print('Create the trainer')
    trainer = CycleGANModelTrainer(model, data_loader.get_trainA_data(), data_loader.get_trainB_data(), data_loader.get_testA_data(), data_loader.get_testB_data(), config, log_dir, checkpoint_dir)

    # print('Start training the model.')
    trainer.train()

if __name__ == '__main__':
    main()