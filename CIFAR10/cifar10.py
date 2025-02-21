import pandas as pd
import os
import torch
import pickle
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, Image
from torchvision import transforms

def get_train_loader(
        args,
):
    
    # load dataset
    if args.load_dataset:
        dataset = load_from_disk(
            os.path.join(args.dataset_dir, "train")
        )

    else:
        dataset = load_dataset(
            'cifar10',
            split="train"
        )
            
    # select CIFAR-10
    with open(args.index_path, 'rb') as handle:
        sub_idx = pickle.load(handle)
    dataset = dataset.select(sub_idx)

    # data augmentation
    augmentations = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )

    ])

    def transform_images(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["img"]]
        labels = examples["label"]
        return {"input": images, "label": labels}

    dataset.set_transform(transform_images)

    # dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.dataloader_num_workers
    )
    
    return train_dataloader


def get_test_loader(
        args,
):
    # load dataset
    if args.load_dataset:
        dataset = load_from_disk(
            os.path.join(args.dataset_dir, "test")
        )

    else:
        dataset = load_dataset(
            'cifar10',
            split="test",
            
        )

    # select CIFAR-10
    with open(args.index_path, 'rb') as handle:
        sub_idx = pickle.load(handle)
    dataset = dataset.select(sub_idx)

    # data augmentation
    augmentations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4942, 0.4851, 0.4504],
            std=[0.2467, 0.2429, 0.2616]
        )
    ])

    def transform_images(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["img"]]
        labels = examples["label"]
        return {"input": images, "label": labels}

    dataset.set_transform(transform_images)

    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.dataloader_num_workers
    )

    return test_dataloader

def get_gen_loader(
        args,
):
    
    df = pd.DataFrame()
    df['path'] = ['{}/{}.png'.format(args.gen_path, i) for i in range(1000)]
    
    dataset = DatasetDict({
        "train": Dataset.from_dict({
            "img": df['path'].tolist(),
    }).cast_column("img", Image()),})
    dataset = dataset["train"]

    ####
    augmentations = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4942, 0.4851, 0.4504],
            std=[0.2467, 0.2429, 0.2616]
        )
        ]
    )

    def transform_images(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["img"]]
        return {"input": images}

    ####
    dataset.set_transform(transform_images)

    gen_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    return gen_dataloader