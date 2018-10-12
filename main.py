import sys
from utils.utils import get_args, process_config, create_dirs
from data_loader.data_loader import DataLoader
from models.cyclegan_attr_model import CycleGANAttrModel
from trainers.cyclegan_trainer import CycleGANModelTrainer

# TODO: Add proper logging

def main():
    # get json configuration filepath from the run argument
    # process the json configuration file
    try:
        args = get_args()
        # TODO: Error if args.config doesn't exist
        config, log_dir, checkpoint_dir = process_config(args.config)
    except:
        print('missing or invalid arguments')
        print('Unexpected error:', sys.exc_info()[0])

    # create the experiment directories
    create_dirs([log_dir, checkpoint_dir])

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