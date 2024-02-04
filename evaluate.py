import json
import os
import torch.utils.data
import torchvision.transforms as transforms
from data_loader import CaptionDataset
from model.DecodeNoAttention import DecoderNoAttention
from model.Decoder import DecoderWithAttention
from model.Resnet101 import Encoder
from util import *

model = 'model/att_model/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'

if __name__ == '__main__':
    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    encoder = Encoder()
    decoder = DecoderWithAttention(attention_dim=attention_dim,
                                   embed_dim=emb_dim,
                                   decoder_dim=decoder_dim,
                                   vocab_size=len(word_map),
                                   dropout=dropout)

    encoder.load_state_dict(torch.load(model)['encoder'])
    try:
        decoder.load_state_dict(torch.load(model)['decoder'])
    except:
        decoder = DecoderNoAttention(attention_dim=attention_dim,
                                     embed_dim=emb_dim,
                                     decoder_dim=decoder_dim,
                                     vocab_size=len(word_map),
                                     dropout=dropout)
        decoder.load_state_dict(torch.load(model)['decoder'])

    encoder = encoder.to('cuda')
    decoder = decoder.to('cuda')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    # test_loader = torch.utils.data.DataLoader(
    #         CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
    #         batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    custom_loader = torch.utils.data.DataLoader(
        CaptionDataset('self_collect/', '', 'CUSTOM', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    evaluate(custom_loader, encoder, decoder, word_map)