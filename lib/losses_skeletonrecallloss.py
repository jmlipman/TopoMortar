from skimage.morphology import skeletonize, dilation
import numpy as np
import torch


def compute_skeletons(target):
    # target.shape = B,C,H,W
    #seg_all = data_dict['segmentation'].numpy()
    seg_all = target.cpu().detach().numpy()
    skels = np.zeros_like(seg_all)

    for b in range(seg_all.shape[0]):
        # Add tubed skeleton GT
        bin_seg = (seg_all[b] > 0)
        #seg_all_skel = np.zeros_like(bin_seg, dtype=np.int16)

        # Skeletonize
        # bin_seg[0] -> Only looking at the first class
        # but in "TopoMortar" is okay because we only have one class..
        if not np.sum(bin_seg[0]) == 0:
            skel = skeletonize(bin_seg[0])
            skel = (skel > 0).astype(np.int16)
            #if self.do_tube: # According to the paper, this is done.
            skel = dilation(dilation(skel))
            skel *= seg_all[b, 0].astype(np.int16)
            skels[b, 0] = skel
    skels = torch.from_numpy(skels)

    return skels
