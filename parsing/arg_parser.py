import os
import shutil
import argparse

DEFAULT_CONFIG_FILE_PATH = os.path.join(".", "configs/config.yaml")
DEFAULT_CHECKPOINT_DIR = os.path.join(".", "checkpoints")
DEFAULT_LOG_DIR = os.path.join(".", "logs")


class ArgParser(object):
    """Parses cmd args."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(add_help=True)
        self._add_args(parser=self.parser)

    @staticmethod
    def _add_args(parser):
        parser.add_argument('-d', '--device-id', type=int, required=True, help="cuda device id")

        parser.add_argument('--config-file-path', default=DEFAULT_CONFIG_FILE_PATH, type=str,
                            help=f"path to config .yaml file (default: {DEFAULT_CONFIG_FILE_PATH})")
        parser.add_argument('--checkpoint-dir', default=DEFAULT_CHECKPOINT_DIR, type=str,
                            help=f"path to checkpoints directory (default: {DEFAULT_CHECKPOINT_DIR})")
        parser.add_argument('--log-dir', default=DEFAULT_LOG_DIR, type=str,
                            help=f"path to log directory (default: {DEFAULT_LOG_DIR})")
        parser.add_argument('--experiment-name', type=str, required=True, help="experiment name")

        parser.add_argument('--seed', default=0, type=int, help="seed for reproducibility (default: 0)")
        parser.add_argument('--num-workers', default=8, type=int, help="number of data loading workers (default: 8)")
        parser.add_argument('--num-epochs', type=int, required=True, help="number of epochs")

        parser.add_argument('--no-log', action='store_true', help="disable logs")
        parser.add_argument('--resume', action='store_true', help="resume training")

    def parse_args(self):
        """Parses cmd args.

        :return: args namespace (argparse)
        """
        args = self.parser.parse_args()
        args = self._adjust_args(args)

        if not args.resume:
            self._make_dirs(args)
            self._copy_config_file(args)

        return args

    @staticmethod
    def _adjust_args(args):
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.experiment_name)
        args.log_dir = os.path.join(args.log_dir, args.experiment_name)

        if args.resume:
            args.config_file_path = os.path.join(args.checkpoint_dir, os.path.split(DEFAULT_CONFIG_FILE_PATH)[-1])

        return args

    @staticmethod
    def _make_dirs(args):
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        if not args.no_log:
            os.makedirs(args.log_dir, exist_ok=True)

    @staticmethod
    def _copy_config_file(args):
        shutil.copy(src=args.config_file_path,
                    dst=os.path.join(args.checkpoint_dir, os.path.split(DEFAULT_CONFIG_FILE_PATH)[-1]))
