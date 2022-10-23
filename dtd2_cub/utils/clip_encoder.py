import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

import clip


class ClipEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        print('ClipEncoder ready.')

    def encode_img(self, img, preprocess=True):
        if type(img) is str:
            img = Image.open(img).convert('RGB')
        if preprocess:
            img = self.preprocess(img)
        image_input = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features[0]

    def encode_img_batch(self, imgs, preprocess=True):
        input_list = list()
        for img in imgs:
            if type(img) is str:
                img = Image.open(img).convert('RGB')
            if preprocess:
                img = self.preprocess(img)
            image_input = img.unsqueeze(0).to(self.device)
            input_list.append(image_input)
        imgs_input = torch.cat(input_list)
        with torch.no_grad():
            image_features = self.model.encode_image(imgs_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_imgs(self, imgs, batch_size=128, preprocess=True):
        img_feats = list()
        while len(imgs) > batch_size:
            imgs_b = imgs[:batch_size]
            img_feats.append(self.encode_img_batch(imgs_b, preprocess))
            imgs = imgs[batch_size:]
            print('%d remaining imgs' % len(imgs))
        if len(imgs) > 0:
            img_feats.append(self.encode_img_batch(imgs))
        img_feats = torch.cat(img_feats)
        return img_feats

    def encode_text_batch(self, text_list):
        tokens = [clip.tokenize(t).to(self.device) for t in text_list]
        text_inputs = torch.cat(tokens).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_text_list(self, text_list, batch_size=128):
        text_feats = list()
        while len(text_list) > batch_size:
            text_b = text_list[:batch_size]
            text_feats.append(self.encode_text_batch(text_b))
            text_list = text_list[batch_size:]
            print('%d remaining texts' % len(text_list))
        if len(text_list) > 0:
            text_feats.append(self.encode_text_batch(text_list))
        return torch.cat(text_feats)


class ClipImgRetriever:
    def __init__(self, imgs=None, img_vecs=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self.img_vecs = img_vecs
        self.imgs = imgs

    def get_img_vecs(self, imgs=None):
        if imgs is None:
            imgs = self.imgs

        img_vecs = list()
        for img in tqdm(imgs, 'get img vecs'):
            with torch.no_grad():
                img_p = self.preprocess(img).unsqueeze(0).to(self.device)
                # print(img_p.shape)
                if img_p.shape[1] != 3:
                    img_vec = torch.zeros_like(img_vecs[-1])
                else:
                    img_vec = self.model.encode_image(img_p)
                    img_vec /= img_vec.norm(dim=-1, keepdim=True).squeeze()
                # img_vec = img_vec.to('cpu')
                img_vecs.append(img_vec)
        return img_vecs

    def __call__(self, desc, img_vecs=None, imgs=None):
        with torch.no_grad():
            desc_tokens = clip.tokenize([desc]).to(self.device)
            phrase_vecs = self.model.encode_text(desc_tokens)
            phrase_vecs /= phrase_vecs.norm(dim=-1, keepdim=True)
            # phrase_vecs.to('cpu')

            if img_vecs is None:
                img_vecs = self.img_vecs
            if imgs is None:
                imgs = self.imgs

            if img_vecs is None:
                img_vecs = self.get_img_vecs(imgs)

            img_scores = list()
            for img_vec in img_vecs:
                img_vec /= img_vec.norm(dim=-1, keepdim=True)
                score = 100.0 * img_vec @ phrase_vecs.T
                score = score.squeeze().to('cpu').item()
                img_scores.append(score)

        return np.asarray(img_scores)
