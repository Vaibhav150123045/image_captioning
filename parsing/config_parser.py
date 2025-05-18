import yaml


class ConfigParser(object):
    """Parses a config file with info about legacy_models, data, etc.

    Args:
        config_file_path (str): path to .yaml file with config.
    """
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path

    def parse_config_file(self):
        """Parses a config file with info about legacy_models, data, etc.

        :return: configs (dict) from config file.
        """
        with open(self.config_file_path, 'r') as config_file:
            config = yaml.safe_load(config_file)

        config_args = dict()

        config_args['model'] = config['model']
        config_args['optimizer'] = config['optimizer']
        config_args['vocabulary'] = config['vocabulary']
        config_args['data'] = config['data']

        return config_args
