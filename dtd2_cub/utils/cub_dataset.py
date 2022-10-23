import os

from dtd2.applications.fine_grained_classification.cub_dataset import CUBDataset as CUB_dtd2


class CUBDataset(CUB_dtd2):
    def __init__(self, data_path='data/CUB_200_2011'):
        super(CUBDataset, self).__init__(data_path=data_path)
        # self.att_phrases = [self.att_name_to_phrase(n) for n in self.att_names]
        # print(self.att_phrases)

    @staticmethod
    def att_name_to_phrase(att_name):
        assert '::' in att_name
        context, adj = att_name.split('::')
        c_words = context.split('_')

        if len(c_words) == 2:  # 'has_size', 'has_shape'
            return adj.split('_')[0] + ' ' + c_words[1]

        if adj.endswith('_tail'):
            return adj.replace('_', ' ')

        if context == 'has_bill_length':
            return 'bill length ' + adj.replace('_', ' ')

        if adj.endswith('-wings'):
            adj = adj[:-len('-wings')]
        words = [adj.split('_')[0]] + c_words[1:-1]
        return ' '.join(words)
