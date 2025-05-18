from parsing.arg_parser import ArgParser
from parsing.config_parser import ConfigParser

from utils import get_device, make_reproducible

from data.vocabulary import Vocabulary
from data.transforms import get_train_transform, get_val_transform
from data.dataset import FlickrDataset
from data.dataloader import get_dataloader

from models.utils import get_model_class

from training.logger import WandBLogger
from training.checkpointer import ModelCheckpointer

from training.trainer import Trainer


def main(args, config):
    make_reproducible(seed=args.seed)

    device = get_device(device_id=args.device_id)

    vocabulary = Vocabulary(captions_file_path=config['vocabulary']['captions_file_path'])

    train_transform = get_train_transform()
    val_transform = get_val_transform()

    train_dataset = FlickrDataset(images_folder_path=config['data']['train']['images_folder_path'],
                                  captions_file_path=config['data']['train']['captions_file_path'],
                                  vocabulary=vocabulary,
                                  transform=train_transform)

    val_dataset = FlickrDataset(images_folder_path=config['data']['val']['images_folder_path'],
                                captions_file_path=config['data']['val']['captions_file_path'],
                                vocabulary=vocabulary,
                                transform=val_transform)

    train_dataloader = get_dataloader(dataset=train_dataset,
                                      batch_size=config['data']['train']['batch_size'],
                                      shuffle=True,
                                      num_workers=args.num_workers)

    val_dataloader = get_dataloader(dataset=val_dataset,
                                    batch_size=config['data']['val']['batch_size'],
                                    shuffle=False,
                                    num_workers=args.num_workers)

    model = get_model_class(model_name=config['model']['name'])(vocabulary=vocabulary, **config['model']['parameters'])
    model.to(device)
    model.freeze()
    optimizer = model.get_optimizer(lr=config['optimizer']['lr'])

    logger = WandBLogger(log_dir=args.log_dir, config=config, resume=args.resume) if not args.no_log else None
    checkpointer = ModelCheckpointer(checkpoint_dir=args.checkpoint_dir, model_config=config['model'])

    last_epoch = 0
    if args.resume:
        last_epoch, model_state_dict, optimizer_state_dict = checkpointer.load_checkpoint(device=device)
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)

    trainer = Trainer(model=model, optimizer=optimizer,
                      checkpointer=checkpointer,
                      logger=logger,
                      device=device,
                      last_epoch=last_epoch)

    trainer.train(train_dataloader=train_dataloader,
                  val_dataloader=val_dataloader,
                  num_epochs=args.num_epochs)


if __name__ == "__main__":
    arg_parser = ArgParser()
    args = arg_parser.parse_args()

    config_parser = ConfigParser(config_file_path=args.config_file_path)
    config = config_parser.parse_config_file()

    main(args=args, config=config)
