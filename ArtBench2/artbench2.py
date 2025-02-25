import os
import pickle
import torch
import pandas as pd
from datasets import Dataset, DatasetDict, Image
from torchvision import transforms
from transformers import PreTrainedTokenizer

def get_train_loader(args, tokenizer: PreTrainedTokenizer):
    
    # Load dataset
    df = pd.read_csv(os.path.join(args.dataset_dir, "ArtBench-10.csv"))
    df["path"] = df.apply(lambda x: os.path.join(args.dataset_dir, "artbench-10-imagefolder", str(x["label"]), x["name"]), axis=1)
    
    with open(args.index_path, 'rb') as handle:
        sub_idx = pickle.load(handle)
    
    dataset = DatasetDict({
        "train": Dataset.from_dict({
            "image": df.loc[sub_idx]['path'].tolist(),
            "label": df.loc[sub_idx]['label'].tolist(),
        }).cast_column("image", Image())
    })
    
    # Define image transformations
    train_transforms = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    # Define caption tokenization
    def tokenize_captions(examples):
        captions = []
        for caption in examples["label"]:
            caption_text = f'a {caption} painting'
            captions.append(caption_text)
        
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return {"input_ids": inputs.input_ids}
    
    # Preprocess function
    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples["image"]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples.update(tokenize_captions(examples))
        return examples
    
    dataset["train"] = dataset["train"].with_transform(preprocess_train)
    
    # Define collate function
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}
    
    # Create DataLoader
    train_dataloader = torch.utils.data.DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
    )
    
    return train_dataloader