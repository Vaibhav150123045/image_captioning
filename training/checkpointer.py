import os
import torch


class ModelCheckpointer(object):
    """Saves and loads .pth.tar checkpoint.

    Args:
        checkpoint_dir (str): path to the folder with checkpoint.
        model_config (dict): model name and parameters.
    """
    def __init__(self, checkpoint_dir, model_config):
        self.checkpoint_dir = checkpoint_dir
        self.model_config = model_config

    def save_checkpoint(self, epoch, model, optimizer):
        """Saves state dicts as a checkpoint in a .pth.tar file.

        Args:
            epoch (int): epoch number.
            model (torch.nn): torch model.
            optimizer (torch.optim): torch optimizer.
        """
        checkpoint = {'epoch': epoch,
                      'model_config': self.model_config,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()}

        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"model.pth.tar"))

    def load_checkpoint(self, device):
        """Loads a checkpoint with state dicts to torch device from a .pth.tar file.

        Args:
            device (torch.device): torch device.

        Returns:
            epoch (int): epoch number.
            model_state_dict (dict): torch model state dict.
            optimizer_state_dict (dict): torch optimizer state dict.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model.pth.tar")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        epoch = checkpoint.get('epoch')
        model_state_dict = checkpoint.get('model_state_dict')
        optimizer_state_dict = checkpoint.get('optimizer_state_dict')

        return epoch, model_state_dict, optimizer_state_dict
