import torch
import numpy as np


def dice_similarity_score(preds, gts):

    with torch.no_grad():

        round_preds = (preds >= 0.5).float() * 1
        denom = torch.sum(round_preds).item() + torch.sum(gts).item()

        return 2*torch.sum(round_preds*gts).item() / denom


def haussdorf_distance(preds, gts):

    from scipy.spatial.distance import directed_hausdorff as directed_hausdorff

    round_preds = (preds >= 0.5).float() * 1
    temp_preds = round_preds.cpu().detach().numpy().reshape(512, 512)
    temp_gts = gts.cpu().detach().numpy().reshape(512, 512)

    dist1 = directed_hausdorff(temp_preds, temp_gts)[0]
    dist2 = directed_hausdorff(temp_gts, temp_preds)[0]

    return max(dist1, dist2)


# borrowed from http://loli.github.io/medpy/_modules/medpy/metric/binary.html
def specificity_LUNA(preds, gts):

    round_preds = (preds >= 0.5).float() * 1
    temp_preds = round_preds.cpu().detach().numpy().reshape(128, 128, 128)
    temp_gts = gts.cpu().detach().numpy().reshape(128, 128, 128)
    
    result = np.atleast_1d(temp_preds.astype(np.bool))
    reference = np.atleast_1d(temp_gts.astype(np.bool))

    tn = np.count_nonzero(~result & ~reference)
    fp = np.count_nonzero(result & ~reference)

    try:
        specificity = tn / float(tn + fp)
    except ZeroDivisionError:
        specificity = 0.0

    return specificity


def recall_sensitivity_LUNA(preds, gts):

    round_preds = (preds >= 0.5).float() * 1
    temp_preds = round_preds.cpu().detach().numpy().reshape(128, 128, 128)
    temp_gts = gts.cpu().detach().numpy().reshape(128, 128, 128)
    
    result = np.atleast_1d(temp_preds.astype(np.bool))
    reference = np.atleast_1d(temp_gts.astype(np.bool))

    tp = np.count_nonzero(result & reference)
    fn = np.count_nonzero(~result & reference)

    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    return recall