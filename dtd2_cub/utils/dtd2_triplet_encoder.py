import os
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image

from dtd2.models.layers.util import print_tensor_stats
from dtd2.models.layers.img_encoder import build_transforms
from dtd2.models.triplet_match.predictors import load_model


class TripletEncoder:
    def __init__(self, trained_path='pretrained_models/dtd2_triplet_bert', model_file='BEST_checkpoint.pth'):
        self.model, self.device = load_model(trained_path, model_file)
        self.model.eval()
        self.img_transform = build_transforms(is_train=False)

    def encode_img(self, img):
        if type(img) is str:
            img = Image.open(img).convert('RGB')

        with torch.no_grad():
            imgs = self.img_transform(img).unsqueeze(0).to(self.device)
            img_vec = self.model.img_encoder(imgs).squeeze()
        return img_vec

    def encode_imgs(self, imgs):
        img_vecs = list()
        for i, img in enumerate(imgs):
            img_vecs.append(self.encode_img(img))
            if i % 100 == 0:
                print('%d imgs encoded' % i)
        print('%d imgs encoded' % len(imgs))
        return torch.stack(img_vecs)

    def encode_text_list(self, text_list, batch_size=128):
        text_feats = list()
        while len(text_list) > batch_size:
            text_b = text_list[:batch_size]
            with torch.no_grad():
                text_feats.append(self.model.lang_encoder(text_b))
            text_list = text_list[batch_size:]
            print('%d remaining texts' % len(text_list))
        if len(text_list) > 0:
            text_feats.append(self.model.lang_encoder(text_list))
        return torch.cat(text_feats)

    def dist(self, img_vec, text_vec):
        return self.model.dist_fn(img_vec, text_vec)
