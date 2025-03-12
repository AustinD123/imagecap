import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from PIL import Image
import requests
from io import BytesIO
import torchvision.transforms as transforms
from datasets import DatasetDict
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MainDataset(Dataset):
    def __init__(self, dataset, tokenizer, transform, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        self.valid_indices = list(range(len(dataset)))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, index):
        item = self.dataset[self.valid_indices[index]]
        sentences = item['sentences']
        text = random.choice(sentences) if isinstance(sentences, list) else sentences
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        try:
            response = requests.get(item['url'], stream=True, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert('RGB')
            img_tensor = self.transform(img)
        except Exception as e:
            # Instead of creating a zero tensor, we raise an exception
            # which will be caught by our custom collate function
            logger.warning(f"Error loading image from URL {item['url']}: {e}")
            raise RuntimeError(f"Failed to load image for index {index}")
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image': img_tensor,
            'imgid': item['imgid'],
            'cocoid': item.get('cocoid', -1),
            'filename': item['filename'],
            'filepath': item['filepath']
        }

def collate_fn(batch):
    # Filter out any None items (failed image loads)
    batch = [item for item in batch if item is not None]
    if not batch:
        # If all items failed, return empty batch with expected structure
        return {
            'input_ids': torch.empty(0, 128),
            'attention_mask': torch.empty(0, 128),
            'image': torch.empty(0, 3, 224, 224),
            'imgid': [],
            'cocoid': [],
            'filename': [],
            'filepath': []
        }
    
    # Combine batch items
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'image': torch.stack([item['image'] for item in batch]),
        'imgid': [item['imgid'] for item in batch],
        'cocoid': [item['cocoid'] for item in batch],
        'filename': [item['filename'] for item in batch],
        'filepath': [item['filepath'] for item in batch]
    }

def create_dataloaders(dataset_dict, tokenizer_name="bert-base-uncased", batch_size=32, 
                       num_workers=4, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    datasets = {}
    dataloaders = {}
    
    for split, dataset in dataset_dict.items():
        datasets[split] = MainDataset(
            dataset=dataset,
            tokenizer=tokenizer,
            transform=transform,
            max_length=max_length
        )
        
        shuffle = (split == 'train' or split == 'restval')
        
        dataloaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn  # Add custom collate function
        )
    
    return dataloaders

def test_dataloader(dataset_dict, batch_size=2, num_workers=0):
    """
    Test if the dataloader works correctly with your dataset
    
    Args:
        dataset_dict: HuggingFace DatasetDict containing your dataset
        batch_size: Small batch size for testing
        num_workers: Number of workers (use 0 for debugging)
    """
    print("Testing dataloader...")
    
    # Create dataloaders with a small batch size
    dataloaders = create_dataloaders(
        dataset_dict=dataset_dict,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Test each split
    for split_name, dataloader in dataloaders.items():
        print(f"\nTesting {split_name} split:")
        print(f"Number of batches: {len(dataloader)}")
        
        # Get the first batch
        try:
            batch = next(iter(dataloader))
            
            # Check batch contents
            print(f"Batch keys: {batch.keys()}")
            print(f"input_ids shape: {batch['input_ids'].shape}")
            print(f"attention_mask shape: {batch['attention_mask'].shape}")
            print(f"image shape: {batch['image'].shape}")
            
            # Print a sample text (decode from input_ids)
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            sample_text = tokenizer.decode(batch['input_ids'][0])
            print(f"Sample text: {sample_text}")
            
            print(f"✅ {split_name} dataloader works!")
        except Exception as e:
            print(f"❌ Error testing {split_name} dataloader: {e}")
    
    print("\nDataloader test complete!")