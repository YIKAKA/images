import json
import numpy as np

def rank(IDs: list, N: int):
    '''
    :param IDs: dataset list of user clicks
    :param N: top N
    :return: most interesting N datasets
    '''
    with open('./id2idx.json', 'r') as f:
        id2idx = json.load(f)  # a dict, map id to idx
    idx2id = {idx: id for id, idx in id2idx.items()}  # a dict, map idx to id
    simi = np.load('./similarity_sub.npy')  # a matrix store similarity

    idxs = [id2idx[ID] for ID in IDs]  #  convert ID to idx
    click_simi = simi[idxs]  # take the similarities
    sum_simi = np.sum(click_simi, axis=0)  # sum
    topNidx = np.argsort(-sum_simi)[:N]  # rank
    return [idx2id[idx] for idx in topNidx]


if __name__ == '__main__':
    IDs = ["011a096e-96fa-43a6-991a-78ddd126a964", "01910f98-2013-42b9-ab2c-ab2f25750896"]
    N = 10
    result = rank(IDs, N)
    print(result)
