import os
import pickle
import torch
import pandas as pd
from datasets import Dataset, DatasetDict, Image
from torchvision import transforms
from transformers import PreTrainedTokenizer
import numpy as np
import random

DATASET_NAME_MAPPING = {
    "artbench": ("image", "label"),
}


def get_train_loader(args, tokenizer: PreTrainedTokenizer, empty_inputs):
    
    # # Load dataset
    # df = pd.read_csv(os.path.join(args.dataset_dir, "ArtBench-10.csv"))
    # df["path"] = df.apply(lambda x: os.path.join(args.dataset_dir, "artbench-10-imagefolder", str(x["label"]), x["name"]), axis=1)
    
    # with open(args.index_path, 'rb') as handle:
    #     sub_idx = pickle.load(handle)
    
    # dataset = DatasetDict({
    #     "train": Dataset.from_dict({
    #         "image": df.loc[sub_idx]['path'].tolist(),
    #         "label": df.loc[sub_idx]['label'].tolist(),
    #     }).cast_column("image", Image())
    # })
    
    # # Define image transformations
    # train_transforms = transforms.Compose([
    #     transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
    #     transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
    #     transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5], [0.5]),
    # ])
    
    # # Define caption tokenization
    # def tokenize_captions(examples):
    #     captions = []
    #     for caption in examples["label"]:
    #         caption_text = f'a {caption} painting'
    #         captions.append(caption_text)
        
    #     inputs = tokenizer(
    #         captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    #     )
    #     return {"input_ids": inputs.input_ids}
    
    # # Preprocess function
    # def preprocess_train(examples):
    #     images = [image.convert("RGB") for image in examples["image"]]
    #     examples["pixel_values"] = [train_transforms(image) for image in images]
    #     examples.update(tokenize_captions(examples))
    #     return examples
    
    # dataset["train"] = dataset["train"].with_transform(preprocess_train)
    
    # # Define collate function
    # def collate_fn(examples):
    #     pixel_values = torch.stack([example["pixel_values"] for example in examples])
    #     pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    #     input_ids = torch.stack([example["input_ids"] for example in examples])
    #     return {"pixel_values": pixel_values, "input_ids": input_ids}
    
    # # Create DataLoader
    # train_dataloader = torch.utils.data.DataLoader(
    #     dataset["train"],
    #     batch_size=args.batch_size,
    #     shuffle=args.shuffle,
    #     num_workers=args.dataloader_num_workers,
    #     collate_fn=collate_fn,
    # )
    
    # return train_dataloader
    import pandas as pd
    df = pd.read_csv(os.path.join(args.dataset_dir, "ArtBench-10.csv"))
    df["path"] = df.apply(lambda x: os.path.join(args.dataset_dir, "artbench-10-imagefolder", str(x["label"]), x["name"]), axis=1)
    # print(df.head())
    
    with open(args.index_path, 'rb') as handle:
        sub_idx = pickle.load(handle)
    print(sub_idx[0:5])
    
    from datasets import DatasetDict, Dataset, load_dataset, Image
    
    dataset = DatasetDict({
        "train": Dataset.from_dict({
            "image": df.loc[sub_idx]['path'].tolist(),
            "label": df.loc[sub_idx]['label'].tolist(),
        }).cast_column("image", Image())
        ,})
    ####
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            ####
            caption = 'a {} painting'.format(caption.replace("_", " ").lower())
            ####
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples


    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
    # Set the training transforms
    ####
    train_dataset = dataset["train"]
    ####
    train_dataset = train_dataset.with_transform(preprocess_train)
    print(len(train_dataset))
    print(train_dataset[0])


    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        
        input_ids_list = []
        for example in examples:
            if np.random.rand()<0.1:
                input_ids_list.append(empty_inputs.input_ids[0])
            else:
                input_ids_list.append(example["input_ids"])

            # input_ids_list.append(example["input_ids"])

        input_ids = torch.stack(input_ids_list)
        
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )