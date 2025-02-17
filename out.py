import torch
from transformers import BertTokenizer
from PIL import Image
from torchvision import transforms
from main_model import fullmodel
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def setup_model(model_path):
    """
    Initialize and load the model and tokenizer
    """
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = fullmodel(embed_size=300, decoder_dim=512, encoder_dim=2048, attention_dim=256)
    
    # Load checkpoint and extract model state
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode and move to device
    model.eval()
    model.to(device)
    
    return model, tokenizer, device

def preprocess_image(image_path):
    """
    Preprocess image for model input
    """
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def generate_caption(model, image_path, tokenizer, device, max_seq_length=20):
    """
    Generate caption for the input image
    """
    # Preprocess and move image to device
    image_tensor = preprocess_image(image_path).to(device)
    
    # Generate features using encoder
    with torch.no_grad():
        features = model.encoder(image_tensor)

    # Initialize decoder with start token
    caption_input = torch.tensor([tokenizer.cls_token_id]).unsqueeze(0).to(device)
    hidden_state, cell_state = model.decoder.inithidden(features)
    caption_generated = []

    # Generate caption word by word
    for _ in range(max_seq_length):
        # Get word embeddings
        embeds = model.decoder.embedding(caption_input)
        
        # Get attention context
        context, _ = model.decoder.attention(features, hidden_state)
        
        # LSTM step
        lstm_input = torch.cat((embeds.squeeze(1), context), dim=1)
        hidden_state, cell_state = model.decoder.lstm_cell(
            lstm_input, 
            (hidden_state, cell_state)
        )
        
        # Generate next word
        output_logits = model.decoder.fcn(hidden_state)
        predicted_word_idx = torch.argmax(output_logits, dim=1).item()
        
        # Stop if end token is generated
        if predicted_word_idx == tokenizer.sep_token_id:
            break
            
        # Add word to caption
        word = tokenizer.decode(predicted_word_idx)
        caption_generated.append(word)
        
        # Update input for next iteration
        caption_input = torch.tensor([[predicted_word_idx]]).to(device)

    return ' '.join(caption_generated)

def main():
    # Configuration
    model_path = "checkpoints/model_epoch_25.pth"
    image_path = "test/1.jpg"
    
    try:
        # Set up model and tokenizer
        model, tokenizer, device = setup_model(model_path)
        
        # Generate and print caption
        caption = generate_caption(model, image_path, tokenizer, device)
        print("Generated Caption:", caption)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()