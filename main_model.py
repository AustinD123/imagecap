import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import BertTokenizer

class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, images):
        features = self.resnet(images)                                     # (batch_size, 2048, 7, 7)
        features = features.permute(0, 2, 3, 1).view(features.size(0), -1, features.size(1)) # (batch, 49, 2048)
        return features

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim)
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim)
        self.attention_proj = nn.Linear(attention_dim, 1)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        encoder_projected = self.encoder_proj(encoder_out)  
        decoder_hidden_projected = self.decoder_proj(decoder_hidden).unsqueeze(1)  
        
        attention = self.relu(encoder_projected + decoder_hidden_projected)
        attention = self.attention_proj(attention).squeeze(2)  
        alpha = self.softmax(attention)  
        
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  
        return attention_weighted_encoding, alpha

class Decoder(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, embed_size, attention_dim, drop):
        super().__init__()
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(BertTokenizer.from_pretrained('bert-base-uncased').vocab_size, embed_size)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim)
        self.fcn = nn.Linear(decoder_dim, self.embedding.num_embeddings)
        self.drop = nn.Dropout(drop)

    def inithidden(self, encoder_out):
        mean_enc = encoder_out.mean(dim=1)
        h, c = self.init_h(mean_enc), self.init_c(mean_enc)
        return h, c

    def forward(self, features, captions):
        embeds = self.embedding(captions)  
        h, c = self.inithidden(features)
        seq_length, batch_size, num_features = captions.size(1), captions.size(0), features.size(1)

        preds = torch.zeros(batch_size, seq_length, self.embedding.num_embeddings).to(features.device)
        alphas = torch.zeros(batch_size, seq_length, num_features).to(features.device)

        for s in range(seq_length):
            context, alpha = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s, :], context), dim=1)  # Correct concatenation
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            preds[:, s, :] = output
            alphas[:, s, :] = alpha

        return preds, alphas

class fullmodel(nn.Module):
    def __init__(self, embed_size, decoder_dim, encoder_dim, attention_dim, drop=0.3):
        super().__init__()
        self.encoder = EncoderCNN()
        self.decoder = Decoder(
            embed_size=embed_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            drop=drop
        )

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs, alphas = self.decoder(features, captions)
        return outputs, alphas
