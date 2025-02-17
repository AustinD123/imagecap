# Image to Caption Project

## Overview
This project aims to generate captions for images using a combination of computer vision and natural language processing techniques. The code provided here is part of the data preprocessing pipeline, which prepares the image and caption data for training a model.

## Code Structure

### `image_caption_dataset.py`
This script defines the `ImageCaptionDataset` class, which is responsible for loading and preprocessing the image-caption pairs. It also includes the necessary transformations for the images and tokenization for the captions.

#### Key Components:
- **ImageCaptionDataset**: A custom PyTorch `Dataset` class that loads images and their corresponding captions, applies transformations, and tokenizes the captions.
- **custom_collate_fn**: A custom collate function to handle padding of captions and stacking of images in a batch.
- **DataLoader**: Creates DataLoader instances for training and testing datasets.

### Usage
1. **Dataset Preparation**:
   - Ensure the image and caption files are correctly placed in the specified directories.
   - The `ImageCaptionDataset` class will automatically load and preprocess the data.

2. **DataLoader**:
   - The `DataLoader` instances (`train_loader` and `test_loader`) are created with a batch size of 32 and use the custom collate function to handle variable-length captions.

3. **Testing**:
   - The script includes a test block to verify the shape of the images and captions loaded by the DataLoader.

### Example Output
```python
Images shape: torch.Size([32, 3, 224, 224])
Captions shape: torch.Size([32, 30])