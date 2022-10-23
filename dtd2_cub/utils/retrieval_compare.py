import os
import random
import numpy as np
from wordcloud import WordCloud

from dtd2.data_api.utils.retrieval_metrics import r_precision, mean_precision_at_k, mean_recall_at_k
from dtd2.data_api.utils.retrieval_metrics import mean_reciprocal_rank, mean_average_precision, average_precision


def compare_pred_to_html(img_path_list, phrase_list, phrase_weight, gt_matrix, method_score_list, output_path, word_cloud=True):
    method1, method2, phrase_ap1, phrase_ap2 = None, None, None, None
    for i, (method, pred_scores) in enumerate(method_score_list):
        print(method)
        i2p_metrics, p2i_metrics = retrieve_eval(gt_matrix, pred_scores)
        method_score_list[i] += [i2p_metrics, p2i_metrics]
        if method.lower() == 'clip':
            method1 = method
            phrase_ap1 = p2i_metrics['query_average_precisions']
        else:
            method2 = method
            phrase_ap2 = p2i_metrics['query_average_precisions']

    os.makedirs(output_path, exist_ok=True)
    if word_cloud:
        pos_neg_phrase_cloud(phrase_list, phrase_weight, method1, method2, phrase_ap1, phrase_ap2, output_path)
    generate_html(img_path_list, phrase_list, gt_matrix, method_score_list, output_path, word_cloud)
    return

def retrieve_eval(gt_matrix, match_scores):
    """
    INPUT:
    gt_matrix: [img_num x phrase_num], binaries showing ground-truth whether img_i and phrase_j matches
    match_scores: [img_num x phrase_num], match_scores[i,j] shows how well img_i and phrase_j matches
    """
    img_num, phrase_num = gt_matrix.shape

    # img to phrase
    # each row is prediction for one image. phrase sorted by pred scores. values are whether the phrase is correct.
    i2p_correct = np.zeros_like(gt_matrix, dtype=bool)  # img_num x phrase_num
    i2p_phrase_idxs = np.zeros_like(i2p_correct, dtype=int)
    for img_i in range(img_num):
        phrase_idx_sorted = np.argsort(-match_scores[img_i, :])
        i2p_phrase_idxs[img_i] = phrase_idx_sorted
        i2p_correct[img_i] = gt_matrix[img_i, phrase_idx_sorted]
    retrieve_binary_lists = i2p_correct
    print('image to phrase')
    i2p_metrics = calculate_metrics(retrieve_binary_lists)

    # phrase to img
    # each row is prediction for one phrase. images sorted by pred scores. values are whether the image is correct.
    p2i_correct = np.zeros_like(gt_matrix, dtype=bool).transpose()  # class_num x img_num
    p2i_img_idxs = np.zeros_like(p2i_correct, dtype=int)
    for pi in range(phrase_num):
        img_idx_sorted = np.argsort(-match_scores[:, pi])
        p2i_img_idxs[pi] = img_idx_sorted
        p2i_correct[pi] = gt_matrix[img_idx_sorted, pi]
    retrieve_binary_lists = p2i_correct
    print('phrase to image')
    p2i_metrics = calculate_metrics(retrieve_binary_lists)
    return i2p_metrics, p2i_metrics


def calculate_metrics(retrieve_binary_lists, verbose=True):
    metrics = dict()
    mean_reciprocal_rank_ = mean_reciprocal_rank(retrieve_binary_lists)
    r_precision_ = r_precision(retrieve_binary_lists)
    mean_average_precision_ = mean_average_precision(retrieve_binary_lists)
    metrics['mean_reciprocal_rank'] = mean_reciprocal_rank_
    metrics['r_precision'] = r_precision_
    metrics['mean_average_precision'] = mean_average_precision_

    metrics['query_average_precisions'] = [average_precision(r) for r in retrieve_binary_lists]

    for k in [1, 5, 10, 20, 50, 100]:
        if k > len(retrieve_binary_lists[0]):
            metrics['precision_at_%03d' % k] = -1
            metrics['recall_at_%03d' % k] = -1
            continue
        precision_at_k_ = mean_precision_at_k(retrieve_binary_lists, k)
        recall_at_k_ = mean_recall_at_k(retrieve_binary_lists, k, gt_count=None)
        metrics['precision_at_%03d' % k] = precision_at_k_
        metrics['recall_at_%03d' % k] = recall_at_k_

    if not verbose:
        return metrics

    # print metrics
    for m, v in sorted(metrics.items(), key=lambda mv: mv[0]):
        if type(v) is list:
            print('%s: [list, skipped]' % m)
        else:
            print('%s: %.4f' % (m, v))
    keys = ['mean_average_precision', 'mean_reciprocal_rank', 'precision_at_005', 'precision_at_020',
            'recall_at_005', 'recall_at_020']
    latex_str = ' & '.join(['%.2f' % (metrics[k] * 100) for k in keys])
    print('latex string')
    print(' & '.join(keys))
    print(latex_str)

    return metrics


def pos_neg_phrase_cloud(phrase_list, phrase_weight, method1, method2, phrase_ap1, phrase_ap2, output_path):
    method_ap = {method1: phrase_ap1, method2: phrase_ap2, 'diff': np.asarray(phrase_ap1) - np.asarray(phrase_ap2)}

    wc = WordCloud(background_color="white", prefer_horizontal=0.9,
                   height=800, width=800, min_font_size=2, margin=2, font_path='output/DIN Alternate Bold.ttf')
    for method, ap in method_ap.items():
        ph_id_sorted = np.argsort(ap)
        # phrase_weight = np.zeros_like(ph_id_sorted)
        # for rank, i in enumerate(ph_id_sorted):
        #     phrase_weight[i] = 1000.0 / (rank + 1)

        if method == 'diff':
            wc.color_func = lambda *args, **kwargs: 'green'
        else:
            wc.color_func = lambda *args, **kwargs: 'blue'
        ph_pos = {phrase_list[i]: np.log(phrase_weight[i]) for i in ph_id_sorted[-80:]}
        # phrase_weight = np.asarray(ap * 100) + 1
        # ph_pos = {phrase_list[i]: phrase_weight[i] for i in ph_id_sorted[-50:]}
        wc.generate_from_frequencies(ph_pos)
        wc.to_file(os.path.join(output_path, '%s_pos_cloud.jpg' % method))

        if method == 'diff':
            wc.color_func = lambda *args, **kwargs: 'orange'
        else:
            wc.color_func = lambda *args, **kwargs: 'red'
        ph_neg = {phrase_list[i]: np.log(phrase_weight[i]) for i in ph_id_sorted[:80]}
        # phrase_weight = 100 - np.asarray(ap * 100)
        # ph_neg = {phrase_list[i]: phrase_weight[i] for i in ph_id_sorted[:50]}
        wc.generate_from_frequencies(ph_neg)
        wc.to_file(os.path.join(output_path, '%s_neg_cloud.jpg' % method))
    return


def generate_html(img_path_list, phrase_list, gt_matrix, method_score_metrics_list, output_path, word_cloud):
    img_num, phrase_num = gt_matrix.shape
    html_str = '<html><body>\n'
    html_str += '<h1>Retrieval comparison </h1>\n'
    html_str += '{img_num} images, {phrase_num} phrases<br><br>\n'.format(img_num=img_num, phrase_num=phrase_num)

    # print metrics
    html_str += '<h2>Metrics</h2>\n'
    latex_keys = ['mean_average_precision', 'mean_reciprocal_rank', 'precision_at_005', 'precision_at_020',
                  'recall_at_005', 'recall_at_020']
    methods = list()
    for method, score, i2p_m, p2i_m in method_score_metrics_list:
        methods.append(method)
        for n, m in [('Phrase', i2p_m), ('Image', p2i_m)]:
            html_str += '<b>%s %s retrieval: </b> <br>\n' % (method, n)
            for k, v in sorted(m.items(), key=lambda mv: mv[0]):
                if type(v) is list:
                    continue
                html_str += '%s: %.4f<br>\n' % (k, v)
            html_str += 'Latex format: %s<br>\n' % ' & '.join(latex_keys)
            latex_str = ' & '.join(['%.2f' % (m[k] * 100) for k in latex_keys])
            html_str += '%s<br><br>\n' % latex_str

    # pos neg phrase clouds
    if word_cloud:
        html_str += '<h2>Phrase clouds</h2>\n'
        for method in methods + ['diff']:
            html_str += '%s pos and neg:<br>\n' % method
            html_str += '<img src="{}" width=300>\n'.format('%s_pos_cloud.jpg' % method)
            html_str += '<img src="{}" width=300>\n<br>'.format('%s_neg_cloud.jpg' % method)
    else:
        html_str += '<h2>Phrase average precision</h2>\n'
        html_str += '%s ; %s; diff<br>\n' % (method_score_metrics_list[0][0], method_score_metrics_list[1][0])
        ap1 = method_score_metrics_list[0][-1]['query_average_precisions']
        ap2 = method_score_metrics_list[1][-1]['query_average_precisions']
        for ph_i, ph in enumerate(phrase_list):
            html_str += '%s: %.2f - %.2f = %.2f<br>\n' \
                        % (ph, ap1[ph_i] * 100, ap2[ph_i] * 100, (ap1[ph_i] - ap2[ph_i]) * 100)

    # img retrieval examples
    top_k = 5
    html_str += '<h2>Image retrieval examples</h2>\n'
    if phrase_num < 100:
        sampled_ph_idxs = range(phrase_num)
    else:
        sampled_ph_idxs = list(range(50)) + random.sample(range(50, phrase_num), 50)
    for phrase_idx in sampled_ph_idxs:
        phrase = phrase_list[phrase_idx]
        html_str += '<hr><h3>Input phrase: {phrase}: ground-truth; {methods} (top 10)</h3>\n'\
            .format(phrase=phrase, methods='; '.join(methods))
        # gt images
        # html_str += '<b> Ground-truth images</b><br>\n'
        gt_img_idxs = list(np.argwhere(gt_matrix[:, phrase_idx]).flatten())
        if len(gt_img_idxs) > top_k:
            gt_img_idxs = random.sample(gt_img_idxs, top_k)
        for img_idx in gt_img_idxs:
            html_str += '<img src={} height=150>\n'.format(img_path_list[img_idx])
        html_str += '<br>\n'

        # plot pred images
        for method, score, _, _ in method_score_metrics_list:
            # html_str += '<br><b> Retrieved top 20 images</b><br>\n'
            img_sorted_idxs = reversed(np.argsort(score[:, phrase_idx].squeeze()))
            for img_idx in img_sorted_idxs[:top_k]:
                html_str += '<img src={} height=150>\n'.format(img_path_list[img_idx])
                # html_str += '<figure style="display: inline-block; margin: 0">' \
                #             '<img src=https://maxwell.cs.umass.edu/mtimm/images/{img_name} ' \
                #             'height=150 border={border}>' \
                #             '<figcaption>{idx}:{score:.3f}</figcaption></figure>\n' \
                #     .format(img_name=img_name, border=border, idx=i + 1, score=pred_score)
            html_str += '<br>\n'

    # phrase retrieval examples
    html_str += '<h2>Phrase retrieval examples</h2>\n'
    vis_img_idxs = random.sample(range(img_num), 100)
    for img_idx in vis_img_idxs:
        html_str += '<img src={} width=300><br>\n'.format(img_path_list[img_idx])

        # gt phrases:
        gt_phrase_idxs = np.argwhere(gt_matrix[img_idx]).flatten()
        html_str += 'gt phrases:<br> %s<br>\n' % ', '.join([phrase_list[pi] for pi in gt_phrase_idxs])

        # pred phrases
        for method, score, _, _ in method_score_metrics_list:
            pred_ph_idxs = reversed(np.argsort(score[img_idx]))
            html_str += '%s Predicted top 20:<br>%s<br>\n' \
                        % (method, ', '.join([phrase_list[pi] for pi in pred_ph_idxs[:20]]))

    html_str += '</body></html>'

    with open(os.path.join(output_path, 'retrieval_comparison.html'), 'w') as f:
        f.write(html_str)

    return

