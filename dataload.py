import torch
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split

import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class ImageCaptionDataset(Dataset):
    def __init__(self, images_path, captions_path, tokenizer, transform=None):
        self.images_path = images_path
        self.tokenizer = tokenizer
        self.transform = transform

        # Load and preprocess captions file
        df = pd.read_csv(captions_path, sep='\t', header=None)
        df[['image_filename', 'caption_id']] = df[0].str.split('#', expand=True)
        df = df.drop(columns=[0])
        df.columns = ['caption', 'image_filename', 'caption_id']

        # Filter out missing images
        self.df = df[df['image_filename'].apply(self.is_valid_image)]

    def is_valid_image(self, image_name):
        """Check if image file exists and return True/False"""
        img_path = os.path.join(self.images_path, image_name.split('.')[0] + '.jpg')
        return os.path.exists(img_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.df.iloc[idx]["image_filename"]
        img_path = os.path.join(self.images_path, image_name.split('.')[0] + '.jpg')

        image = Image.open(img_path).convert("RGB")
        caption = self.df.iloc[idx]["caption"]

        if self.transform:
            image = self.transform(image)

        caption_tokens = self.tokenizer(
            caption, padding='max_length', max_length=30, truncation=True, return_tensors="pt"
        )
        caption_tensor = caption_tokens['input_ids'].squeeze(0)

        return image, caption_tensor


def custom_collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions

# transformations for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define paths
images_path = r"C:\Users\austi\Downloads\Flickr8k_Dataset\Flicker8k_Dataset"
captions_path = r"C:\Users\austi\Downloads\Flickr8k_text\Flickr8k.lemma.token.txt"

# Create the dataset
dataset = ImageCaptionDataset(images_path=images_path, captions_path=captions_path, tokenizer=tokenizer, transform=transform)

# Split the full dataset into training and testing sets
indices = list(range(len(dataset)))  # Use the full dataset
train_size = int(0.8 * len(indices))  # 80% for training
test_size = len(indices) - train_size  # 20% for testing

train_indices, test_indices = train_test_split(indices, train_size=train_size, test_size=test_size, random_state=42)

# Create subsets for training and testing
train_subset = Subset(dataset, train_indices)
test_subset = Subset(dataset, test_indices)

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
import os
# Test the DataLoader
if __name__ == '__main__':
    for images, captions in train_loader:
        print("Images shape:", images.shape)  # Should be (batch_size, 3, 224, 224)
        print("Captions shape:", captions.shape)  # Should be (batch_size, max_sequence_length)
        images_path = r"C:\Users\austi\Downloads\Flickr8k_Dataset\Flicker8k_Dataset"
        captions_path = r"C:\Users\austi\Downloads\Flickr8k_text\Flickr8k.lemma.token.txt"

        # Check if directories exist
        print("Image folder exists:", os.path.exists(images_path))
        print("Captions file exists:", os.path.exists(captions_path))
        break



