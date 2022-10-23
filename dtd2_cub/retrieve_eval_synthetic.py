import os
import numpy as np
import torch
from tqdm import tqdm

import clip

from dtd2.applications.synthetic_imgs.dataset import SyntheticData
from dtd2.applications.synthetic_imgs.predict_retrieve \
    import make_input_cases, retrieve_img_eval, retrieve_img_visualize


form_sentence = False
print('form_sentence:', form_sentence)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print('model ready.')

dataset = SyntheticData()
print('dataset ready.')

img_vecs = list()
for i in tqdm(range(len(dataset)), desc='encoding images'):
    img_i, c1_i, c2_i = dataset.unravel_index(i)
    img = dataset.get_img(img_i, c1_i, c2_i, convert_rgb=False)
    with torch.no_grad():
        img_p = preprocess(img).unsqueeze(0).to(device)
        img_vec = model.encode_image(img_p)
        img_vec /= img_vec.norm(dim=-1, keepdim=True)
    img_vecs.append(img_vec)

print('img_vecs ready.')

def clip_retrieve_fn(desc):
    if form_sentence:
        desc = 'an image with %s' % desc
    with torch.no_grad():
        phrases = clip.tokenize([desc]).to(device)
        phrase_vecs = model.encode_text(phrases)
        phrase_vecs /= phrase_vecs.norm(dim=-1, keepdim=True)

    img_scores = [0] * len(img_vecs)
    for i, img_vec in enumerate(img_vecs):
        img_scores[i] = (100.0 * img_vec @ phrase_vecs.T)  # .softmax(dim=-1)

    return np.asarray(img_scores)

for exp_name in ['fore_color', 'back_color', 'color_pattern', 'two_colors']:
    input_cases = make_input_cases(dataset, exp_name)
    print('\n%s: #input %d, #pos %d, pos_rate %.4f'
          % (exp_name, len(input_cases), np.mean([len(ic[1]) for ic in input_cases]),
             np.mean([len(ic[1]) / (len(ic[1]) + len(ic[2])) for ic in input_cases])))

    clip_results = retrieve_img_eval(clip_retrieve_fn, input_cases)
    print('clip_retrieve done. acc_all: %.4f +- %.4f; acc_hard: %.4f +- %.4f'
          % (float(np.mean(clip_results[1])), float(np.std(clip_results[1])),
             float(np.mean(clip_results[2])), float(np.std(clip_results[2]))))
    # results = None
    results = {'input_cases': input_cases,
               'clip_results': clip_results}

    result_path = 'models/clip/retrieve_synthetic_results'
    os.makedirs(result_path, exist_ok=True)
    np.save(os.path.join(result_path, 'results_%s.npy' % exp_name), results, allow_pickle=True)

    retrieve_img_visualize(results, exp_name, dataset, model_names=('clip_results',), html_folder=result_path)
    retrieve_img_visualize(results, exp_name, dataset, model_names=('clip_results',), html_folder=result_path,
                           hard_only=True)



