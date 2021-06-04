import argparse
import configparser

from train import train


def preprocess_config(c):
    conf_dict = {}
    int_params = ['data.gpu', 'data.cuda', 'model.latent_dim', 'train.epochs',
                  'train.patience', 'train.seed', 'train.shuffle_buffer',
                  'train.batch_size']
    float_params = ['train.lr']
    int_list_params = ['data.shape', 'model.layers']
    for param in c:
        if param in int_params:
            conf_dict[param] = int(c[param])
        elif param in float_params:
            conf_dict[param] = float(c[param])
        elif param in int_list_params:
            conf_dict[param] = [int(val) for val in c[param].split(',')]

        else:
            conf_dict[param] = c[param]
    return conf_dict


parser = argparse.ArgumentParser(description='Run training')
parser.add_argument("--config", type=str, default="./Configs/default.INI",
                    help="Path to the configuration file.")
parser.add_argument("-v", dest="verbose", action='store_true', default=False,
                    help="Print traces/Verbose.")

# Run training
args = vars(parser.parse_args())
dbg = args['verbose']
config = configparser.ConfigParser()
config.read(args['config'])
config = preprocess_config(config['TRAIN'])
train(config)
