
import os
import argparse
import configparser
import glob
from Preprocessors import split_train_test, split_OK_NOK
from Preprocessors import Augmentor

def preprocess_config(c):
    conf_dict = {}
    int_params = ['data.split.seed', 'augm.times']
    float_params = ['data.split.frac']
    int_list_params = ['data.shape']
    type = None
    try:
        for param in c:
            if param in int_params:
                type='int'
                conf_dict[param] = int(c[param])
            elif param in int_list_params:
                type='int list'
                conf_dict[param] = [int(val) for val in c[param].split(',')]
            elif param in float_params:
                type='float'
                conf_dict[param] = float(c[param])
            else:
                type='str'
                conf_dict[param] = c[param]
    except ValueError:
        print(f'{param} should be {type}, but it couldn\'t be casted. '\
                'Try fixing the .INI config file')
        exit(2)
    return conf_dict

def get_filenames(path, img_format):
    return sorted(glob.glob(os.path.join(path, f'*.{img_format}')))
    

# Read config file and setup env
parser = argparse.ArgumentParser(description='Splits and augments dataset')
parser.add_argument("--config", type=str, default="./Configs/default.INI",
                    help="Path to the config file.")
parser.add_argument("-v", dest="verbose", action='store_true', default=False,
                    help="Print traces/Verbose.")

# Read stdin
args = vars(parser.parse_args())
verb = args['verbose']
# Read config
if verb: print('Reading configuration file...')
config = configparser.ConfigParser()
config.read(args['config'])
config = preprocess_config(config['PREPROCESSING'])

# Read image filenames

# call splitter
if verb: print('\nSplitting files into train and test for autoencoder...')
ok_files, nok_files = split_OK_NOK(config['data.anots'])
split = split_train_test(ok_files, fraction=config['data.split.frac'],
                         seed=config['data.split.seed'])
# Append NOK files to dictionary
split['NOK'] = nok_files

if verb: print('\nCarrying on data augmentation...')
# setup augmentator
augment_layer = Augmentor(config['data.out'], config['data.shape'])
# call augmentator
augment_layer.augmentate(config['augm.times'], config['data.input'],
                         split, config['data.format'])

# finish
if verb: print('\nYou\'ve just got the data augmented. You\'re welcome!')



