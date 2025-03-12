# Import necessary libraries
from datasets import load_dataset, DatasetDict
from dataloader import test_dataloader  # Import from wherever you saved the code

from datasets import load_dataset

dataset_dict = load_dataset("yerevann/coco-karpathy")
# 1. First, let's check if your dataset is properly loaded
print("Dataset information:")
print(f"Dataset type: {type(dataset_dict)}")
print(f"Available splits: {list(dataset_dict.keys())}")
print(f"Train split size: {len(dataset_dict['train'])}")

# 2. Test with a single example first
print("\nTesting with a single example:")
sample_item = dataset_dict['train'][0]
print("Sample item keys:", sample_item.keys())
print("Sample sentences:", sample_item['sentences'][:2], "...")  # Show first 2 sentences

# 3. Run the test_dataloader function to verify everything works
test_dataloader(
    dataset_dict=dataset_dict,
    batch_size=2,  # Small batch size for testing
    num_workers=0  # No multi-processing for easier debugging
)