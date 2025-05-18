import torchvision.transforms


def get_train_transform():
    return torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256)),
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                            std=(0.229, 0.224, 0.225))])


def get_val_transform():
    return torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256)),
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                            std=(0.229, 0.224, 0.225))])
