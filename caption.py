import json
import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from config import emb_dim, attention_dim, decoder_dim, dropout
from model.DecodeNoAttention import DecoderNoAttention
from model.Decoder import DecoderWithAttention
from model.Resnet101 import Encoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 008, 492, 488
img_path = 'self_collect/img/492.jpg'
word_map_path = 'coco/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'
model_path = 'model/att_model/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'


def generate_caption(image, encoder, decoder, word_map, max_len=100):
    """
    Generate a caption for a given image using the trained encoder and decoder models.
    :param image: input image tensor of shape (3, 256, 256)
    :param encoder: trained encoder model
    :param decoder: trained decoder model
    :param word_map: word-to-index mapping
    :param max_len: maximum length of the generated caption
    :return: generated caption as a string
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.eval()
    decoder.eval()

    image = image.transpose(2, 0, 1)
    image = (torch.FloatTensor(image) / 255.).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image = normalize(image)  # (3, 256, 256)
    image = image.unsqueeze(0)  # Add a batch dimension
    reversed_word_map = {value: key for key, value in word_map.items()}

    with torch.no_grad():
        # Encode
        encoder_out = encoder(image)
        encoder_dim = encoder_out.size(3)
        batch_size = encoder_out.size(0)

        # Flatten encoding
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)

        # Initialize LSTM
        prev_word = torch.LongTensor([[word_map['<start>']]] * batch_size).to(device)
        seqs = prev_word
        h, c = decoder.init_hidden_state(encoder_out)

        # Generate caption
        for t in range(max_len):
            if hasattr(decoder, 'attention'):
                attention_weighted_encoding, _ = decoder.attention(encoder_out, h)
                gate = decoder.sigmoid(decoder.f_beta(h))
                attention_weighted_encoding = gate * attention_weighted_encoding
                h, c = decoder.decode_step(
                    torch.cat([decoder.embedding(prev_word.squeeze(1)), attention_weighted_encoding], dim=1),
                    (h, c)
                )
            else:
                gate = decoder.sigmoid(decoder.f_beta(h))
                h, c = decoder.decode_step(
                    torch.cat([decoder.embedding(prev_word.squeeze(1)), gate], dim=1),
                    (h, c)
                )

            scores = decoder.fc(h)
            prev_word = scores.max(1)[1].unsqueeze(1)
            seqs = torch.cat([seqs, prev_word], dim=1)

            # Check if the end token is generated
            if prev_word.item() == word_map['<end>']:
                break

        # Convert the generated caption tensor to a list of word indices
        caption = seqs.squeeze().tolist()
        caption = [reversed_word_map[word_idx] for word_idx in caption]

        # Remove special tokens from the generated caption
        caption = [word for word in caption if word not in ['<start>', '<end>', '<pad>']]

        # Convert the list of words to a string
        generated_caption = ' '.join(caption)

        return generated_caption

if __name__ == '__main__':
    with open(word_map_path, 'r') as j:
        word_map = json.load(j)

    encoder = Encoder()
    decoder = DecoderWithAttention(attention_dim=attention_dim,
                                   embed_dim=emb_dim,
                                   decoder_dim=decoder_dim,
                                   vocab_size=len(word_map),
                                   dropout=dropout)

    encoder.load_state_dict(torch.load(model_path)['encoder'])
    try:
        decoder.load_state_dict(torch.load(model_path)['decoder'])
    except:
        decoder = DecoderNoAttention(attention_dim=attention_dim,
                                     embed_dim=emb_dim,
                                     decoder_dim=decoder_dim,
                                     vocab_size=len(word_map),
                                     dropout=dropout)
        decoder.load_state_dict(torch.load(model_path)['decoder'])
    encoder = encoder.to('cuda')
    decoder = decoder.to('cuda')
    img = load_img(img_path, target_size=(256, 256))
    img = img_to_array(img)
    cap = generate_caption(img, encoder, decoder, word_map)
    # plt.figure(figsize=(8, 8))
    plt.imshow(img / 255)
    plt.axis('off')
    plt.title(cap)
    plt.show()


