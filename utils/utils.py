import argparse
import json
import os
import time

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        required=True,
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args

def get_config_from_json(json_file):
    """Get the config from a json file

    Args:
        json_file (str): path

    Returns:
        config (dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    return config_dict

def process_config(json_file):
    config_dict = get_config_from_json(json_file)
    return config_dict

def create_dirs(config):
    """Create directories if not found

    Args:
        dirs (str): directories

    Returns:
        exit_code: 0:success -1:failed
    """
    log_dir = os.path.join('experiments', time.strftime("%Y-%m-%d/",time.localtime()), config['exp_name'], 'logs/')
    checkpoint_dir = os.path.join('experiments', time.strftime("%Y-%m-%d/",time.localtime()), config['exp_name'], 'checkpoints/')
    try:
        for dir_ in [log_dir, checkpoint_dir]:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)
    return log_dir, checkpoint_dir