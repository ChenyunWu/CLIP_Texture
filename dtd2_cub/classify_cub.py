import numpy as np
from tqdm import tqdm
import torch
import random

import clip
from dtd2.applications.fine_grained_classification.cub_dataset import CUBDataset


def cub_atts_to_desc(bird_name, att_names):
    if bird_name is None:
        bird_name = 'bird'
    # if bird_name.lower()[-4:] != 'bird':
    #     bird_name += ' bird'

    if len(att_names) == 0:
         return 'An image of a %s.' % bird_name

    part_adj = dict()
    primary_color = None
    for att_name in att_names:
        assert '::' in att_name
        part, adj = att_name.split('::')
        part_words = part.split('_')

        if adj == primary_color:
            continue

        if adj == 'curved_(up_or_down)':
            adj = 'curved'

        if len(part_words) == 2:  # 'has_size', 'has_shape'
            part = 'bird'
            if '(' in adj:
                adj = adj.split('_(')[0]
            adj += ' ' + part_words[1]

        elif 'primary' in part:
            part = 'bird'

        elif adj.endswith('_tail'):
            part = 'tail'
            adj = adj.split('_')[0]

        elif part == 'has_bill_length':
            part = 'bill'
            if 'long' in adj:
                adj = 'long'
            elif 'short' in adj:
                adj = 'short'
            else:
                continue

        elif adj.endswith('-wings'):
            part = 'wings'
            adj = adj[:-len('-wings')]

        else:
            part = ' '.join(part_words[1:-1])

        adj = adj.replace('_', ' ')
        if part in part_adj:
            part_adj[part].append(adj)
        else:
            part_adj[part] = [adj]

    bird_adjs = part_adj.get('bird', [])
    if len(bird_adjs) > 0:
        bird_name = ' '.join(bird_adjs) + ' ' + bird_name
    phrases = list()
    for part, adjs in part_adj.items():
        if part == 'bird':
            continue
        phrase = ' '.join(adjs) + ' ' + part
        phrases.append(phrase)

    if len(phrases) == 0:
        return 'An image of a %s.' % bird_name

    return 'An image of a %s with %s.' % (bird_name, ', '.join(phrases))


def cub_classify(cub_dataset=None, model=None, preprocess=None, device=None, class_text_mode='both',
                 att_num=40, att_thresh=10, class_level=0):
    if cub_dataset is None:
        cub_dataset = CUBDataset(data_path='data/CUB_200_2011')
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        print('model ready.')
    print('att_num', att_num)
    print('att_thresh', att_thresh)
    if class_text_mode == 'name':
        bird_names = cub_dataset.taxonomy[0]
        text_inputs = torch.cat([clip.tokenize(f"An image of a {bn}") for bn in bird_names]).to(device)
    else:  # att
        sentences = list()
        for ci, att_labels in enumerate(cub_dataset.class_att_labels):
            att_scores = att_labels - np.mean(cub_dataset.class_att_labels, axis=0)
            att_idxs = list(reversed(np.argsort(att_scores)))
            atts = [cub_dataset.att_names[att_i] for att_i in att_idxs[:att_num] if att_scores[att_i] > att_thresh]
            if class_text_mode == 'both':
                s = cub_atts_to_desc(bird_name=cub_dataset.taxonomy[class_level][ci], att_names=atts)
            else:
                s = cub_atts_to_desc(bird_name=None, att_names=atts)
            # if len(s) > 250:
            #     s = s[:250]
            sentences.append(s)
        for s in sentences[:10]:
            print(s)
        text_inputs = torch.cat([clip.tokenize(s) for s in sentences]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    predicts = dict()

    # for img_idx in tqdm(random.sample(cub_dataset.img_splits['test'], 500)):
    for img_idx in tqdm(cub_dataset.img_splits['test']):
        img_data = cub_dataset.img_data_list[img_idx]
        img = cub_dataset.load_img(img_idx)
        gt_label = img_data['class_label']

        image_input = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(5)
            gt_rank = torch.sum(similarity[0] > similarity[0, gt_label])
        predicts[img_idx] = {'top_5_labels': indices.detach().cpu().numpy(),
                             'top_5_probs': values.detach().cpu().numpy(),
                             'gt_rank': gt_rank.detach().cpu().numpy(),
                             'gt_label': gt_label}

    return evaluate(predicts)


def evaluate(predicts):
    count = len(predicts)
    metrics = {'mrr': 0, 'acc_top1': 0, 'acc_top5': 0, 'acc_top10': 0}
    class_count = np.zeros(200)
    class_correct = np.zeros(200)
    for p in predicts.values():
        r = p['gt_rank']
        metrics['mrr'] += 1.0 / (1 + r)
        metrics['acc_top1'] += int(r == 0)
        metrics['acc_top5'] += int(r < 5)
        metrics['acc_top10'] += int(r < 10)

        ci = p['gt_label']
        class_count[ci] += 1
        if r == 0:
            class_correct += 1

    for k, v in metrics.items():
        metrics[k] = v / count
        print(k, metrics[k])

    metrics['class_acc'] = list(class_correct / class_count)
    return metrics


if __name__ == '__main__':
    cub_dataset = CUBDataset(data_path='data/CUB_200_2011')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    print('model ready.')
    cub_classify(cub_dataset=cub_dataset, model=model, preprocess=preprocess, device=device,
                 class_text_mode='att', att_num=15, att_thresh=20)

    # for att_num in [3, 5, 10, 15]:
    # for th in [40, 20, 60]:
    # for class_level in range(1, 5):
    # th = 20
    # for class_level in [0]:
    #     for att_num in [15]:
    #         # if class_level == 0 and att_num > 5:
    #         #     continue
    #         print('\nth %d; att_num %d; level %d' % (th, att_num, class_level))
    #         name_count = len(set(cub_dataset.taxonomy[class_level]))
    #         print('%d unique names; acc upperbound %.3f' % (name_count, name_count / len(cub_dataset.taxonomy[0])))
    #         cub_classify(cub_dataset=cub_dataset, model=model, preprocess=preprocess, device=device,
    #                      att_num=att_num, att_thresh=th, class_level=class_level)


