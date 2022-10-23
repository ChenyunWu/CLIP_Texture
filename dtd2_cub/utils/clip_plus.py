import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import time
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import clip
from dtd2.models.triplet_match.dataset import TripletTrainData
from utils.clip_encoder import ClipEncoder


class ClipPlusEncoder:
    def __init__(self, clip_encoder=None, clip_vec_dim=512, out_vec_dim=512, distance='l2_s', load_path=None):
        self.clip_encoder = clip_encoder
        if clip_encoder is None:
            self.clip_encoder = ClipEncoder()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.img_layer = nn.Linear(clip_vec_dim, out_vec_dim)
        self.text_layer = nn.Linear(clip_vec_dim, out_vec_dim)
        self.img_layer.to(self.device)
        self.text_layer.to(self.device)

        if load_path is not None:
            self.load(load_path)

        if distance == 'l2_s':
            self.dist_fn = lambda v1, v2: (v1 - v2).pow(2).sum(dim=-1)
        elif distance == 'l2':
            self.dist_fn = lambda v1, v2: (v1 - v2).pow(2).sum(dim=-1).sqrt()
        elif distance == 'cos':
            self.dist_fn = lambda v1, v2: 1.0 - nn.functional.cosine_similarity(v1, v2, dim=0)
        else:
            raise NotImplementedError

        print('CLIPP ready')
        return

    def train(self):
        self.img_layer.train()
        self.text_layer.train()

    def eval(self):
        self.img_layer.eval()
        self.text_layer.eval()

    def save(self, path):
        torch.save({'img_layer': self.img_layer.state_dict(),
                    'text_layer': self.text_layer.state_dict()}, path)
        return

    def load(self, path):
        checkpoint = torch.load(path)
        self.img_layer.load_state_dict(checkpoint['img_layer'])
        self.text_layer.load_state_dict(checkpoint['text_layer'])
        return

    def encode_imgs(self, imgs, batch_size=128, preprocess=True):
        feats = list()
        while len(imgs) > batch_size:
            clip_feats = self.clip_encoder.encode_img_batch(imgs[:batch_size], preprocess)
            feats.append(self.img_layer(clip_feats.to(torch.float)))
            imgs = imgs[batch_size:]
            print('%d remaining imgs' % len(imgs))
        if len(imgs) > 0:
            clip_feats = self.clip_encoder.encode_img_batch(imgs, preprocess)
            feats.append(self.img_layer(clip_feats.to(torch.float)))
        feats = torch.cat(feats)
        return feats

    def encode_text_list(self, text_list, batch_size=128):
        clip_feats = self.clip_encoder.encode_text_list(text_list, batch_size)
        feats = list()
        while len(clip_feats) > batch_size:
            feats_b = clip_feats[:batch_size]
            feats.append(self.text_layer(feats_b.to(torch.float)))
            clip_feats = clip_feats[batch_size:]
            print('%d remaining texts' % len(clip_feats))
        if len(clip_feats) > 0:
            feats.append(self.text_layer(clip_feats.to(torch.float)))
        feats = torch.cat(feats)
        return feats


def train(clipp_encoder=None, init_lr=0.0001, batch_size=128, max_epoch=200,
          prompt_format='An image of {} texture.'):
    print('train clip_plus init_lr=%f' % init_lr)
    if clipp_encoder is None:
        clipp_encoder = ClipPlusEncoder()
    clipp_encoder.train()

    dataset = TripletTrainData(split='train', lang_input='description', neg_img=True, neg_lang=True,
                               img_transform=clipp_encoder.clip_encoder.preprocess)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

    optimizer = torch.optim.Adam(params=itertools.chain(clipp_encoder.img_layer.parameters(),
                                                        clipp_encoder.text_layer.parameters()), lr=init_lr)

    output_path = 'output/clipp/models/prompt_lr{}/'.format(init_lr)
    os.makedirs(output_path, exist_ok=True)

    # training loop
    step = 1
    epoch = 0
    epoch_float = 0.0
    epoch_per_step = batch_size * 1.0 / len(dataset)
    best_eval_metric = 0
    best_metrics = None
    best_eval_count = 0
    early_stop = False
    neg_margin = 1.0
    while epoch <= max_epoch and not early_stop:
        for pos_imgs, pos_phrases, neg_imgs, neg_phrases in data_loader:
            pos_imgs = pos_imgs.to('cuda')
            neg_imgs = neg_imgs.to('cuda')
            pos_sents = [prompt_format.format(ph) for ph in pos_phrases]
            neg_sents = [prompt_format.format(ph) for ph in neg_phrases]

            pos_img_vecs = clipp_encoder.encode_imgs(pos_imgs, batch_size, preprocess=False)
            pos_sent_vecs = clipp_encoder.encode_text_list(pos_sents, batch_size)
            pos_dist = clipp_encoder.dist_fn(pos_img_vecs, pos_sent_vecs)

            neg_img_vecs = clipp_encoder.encode_imgs(neg_imgs, batch_size, preprocess=False)
            neg_img_dist = clipp_encoder.dist_fn(pos_sent_vecs, neg_img_vecs)
            neg_img_losses = torch.relu(pos_dist - neg_img_dist + neg_margin)
            neg_img_loss = torch.mean(neg_img_losses)

            neg_sent_vecs = clipp_encoder.encode_text_list(neg_sents, batch_size)
            neg_sent_dist = clipp_encoder.dist_fn(pos_img_vecs, neg_sent_vecs)
            neg_sent_losses = torch.relu(pos_dist - neg_sent_dist + neg_margin)
            neg_sent_loss = torch.mean(neg_sent_losses)

            loss = neg_img_loss + neg_sent_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step <= 5 or step % 50 == 0:
                img_acc = torch.sum(neg_img_dist > pos_dist) * 1.0 / batch_size
                sent_acc = torch.sum(neg_sent_dist > pos_dist) * 1.0 / batch_size
                print('[%s] epoch-%d step-%d: loss %.4f (neg_img: %.4f, neg_sent %.4f); img_acc %.2f, sent_acc %.2f'
                      % (time.strftime('%m/%d %H:%M:%S'), epoch, step, loss, neg_img_loss, neg_sent_loss,
                         img_acc * 100.0, sent_acc * 100.0))

            step += 1
            epoch_float += epoch_per_step

        if epoch % 10 == 0:
            clipp_encoder.save(os.path.join(output_path, 'clipp_epoch{}.pth'.format(epoch)))

        epoch += 1
    return clipp_encoder


if __name__ == '__main__':
    clipp_encoder = ClipPlusEncoder()
    train(init_lr=0.01)
