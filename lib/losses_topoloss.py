#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Xiaoling Hu
# Created Date: Tue June 22 9:00:00 PDT 2021
# Edited By: J. Miguel Valverde
# Edited Date: November 2023
# =============================================================================

import time
import numpy as np
import gudhi as gd
import torch
from torch.nn.modules.loss import _Loss
import math


def compute_dgm_force(lh_dgm, gt_dgm, pers_thd=0.03, pers_thd_perfect=0.99, do_return_perfect=False):
    """
    Compute the persistent diagram of the image

    Args:
        lh_dgm: likelihood persistent diagram.
        gt_dgm: ground truth persistent diagram.
        pers_thd: Persistent threshold, which also called dynamic value, which measure the difference.
        between the local maximum critical point value with its neighouboring minimum critical point value.
        The value smaller than the persistent threshold should be filtered. Default: 0.03
        pers_thd_perfect: The distance difference between two critical points that can be considered as
        correct match. Default: 0.99
        do_return_perfect: Return the persistent point or not from the matching. Default: False

    Returns:
        force_list: The matching between the likelihood and ground truth persistent diagram
        idx_holes_to_fix: The index of persistent points that requires to fix in the following training process
        idx_holes_to_remove: The index of persistent points that require to remove for the following training
        process

    """
    lh_pers = abs(lh_dgm[:, 1] - lh_dgm[:, 0])
    if (gt_dgm.shape[0] == 0):
        gt_pers = None;
        gt_n_holes = 0;
    else:
        gt_pers = gt_dgm[:, 1] - gt_dgm[:, 0]
        gt_n_holes = len(gt_pers)  # number of holes in gt

    if (gt_pers is None or gt_n_holes == 0):
        idx_holes_to_fix = []
        idx_holes_to_remove = list(range(len(lh_pers)))
        idx_holes_perfect = []
    else:
        # check to ensure that all gt dots have persistence 1
        #tmp = gt_pers > pers_thd_perfect

        # get "perfect holes" - holes which do not need to be fixed, i.e., find top
        # lh_n_holes_perfect indices
        # check to ensure that at least one dot has persistence 1; it is the hole
        # formed by the padded boundary
        # if no hole is ~1 (ie >.999) then just take all holes with max values
        # Miguel: they're perfect because the diff between death and birth is
        # near 1, i.e., they appear all the time regardless of the thr.
        tmp = lh_pers > pers_thd_perfect  # old: assert tmp.sum() >= 1
        lh_pers_sorted_indices = np.argsort(lh_pers)[::-1]
        if np.sum(tmp) >= 1:
            lh_n_holes_perfect = tmp.sum()
            idx_holes_perfect = lh_pers_sorted_indices[:lh_n_holes_perfect];
        else:
            idx_holes_perfect = []

        # find top gt_n_holes indices
        # Miguel: so, this makes it focus on the holes that are near perfect
        # Miguel: I thought that this method maps the persistence maps from GT
        # and likelihood, but it seems that it determines which holes to fix
        # based not on the which holes are similar to the ground truth but on
        # the "holes" that are close to perfect. The problem of this is that
        # it can try to fix a near-to-perfect hole that is wrong... I guess that
        # in practice this is avoided because the predictions are assumed to be
        # quite good, that's why first the CE loss needs to be minimized, and,
        # after a while, topoloss needs to be used.
        idx_holes_to_fix_or_perfect = lh_pers_sorted_indices[:gt_n_holes];

        # the difference is holes to be fixed
        idx_holes_to_fix = list(
            set(idx_holes_to_fix_or_perfect) - set(idx_holes_perfect))

        # remaining holes are all to be removed
        idx_holes_to_remove = lh_pers_sorted_indices[gt_n_holes:];

    # only select the ones whose persistence is large enough
    # set a threshold to remove meaningless persistence dots
    # Miguel: because if the persistence is too low, then we just don't care
    idx_valid = np.where(lh_pers > pers_thd)[0]
    idx_holes_to_remove = list(
        set(idx_holes_to_remove).intersection(set(idx_valid)))

    force_list = np.zeros(lh_dgm.shape)

    # push each hole-to-fix to (0,1)
    force_list[idx_holes_to_fix, 0] = 0 - lh_dgm[idx_holes_to_fix, 0]
    force_list[idx_holes_to_fix, 1] = 1 - lh_dgm[idx_holes_to_fix, 1]

    # push each hole-to-remove to (0,1)
    force_list[idx_holes_to_remove, 0] = lh_pers[idx_holes_to_remove] / \
                                         math.sqrt(2.0)
    force_list[idx_holes_to_remove, 1] = -lh_pers[idx_holes_to_remove] / \
                                         math.sqrt(2.0)

    if (do_return_perfect):
        return force_list, idx_holes_to_fix, idx_holes_to_remove, idx_holes_perfect

    return force_list, idx_holes_to_fix, idx_holes_to_remove

def getCriticalPoints(likelihood):
    """
    Compute the critical points of the image (Value range from 0 -> 1)

    Args:
        likelihood: Likelihood image from the output of the neural networks

    Returns:
        pd_lh:  persistence diagram.
        bcp_lh: Birth critical points.
        dcp_lh: Death critical points.
        Bool:   Skip the process if number of matching pairs is zero.

    """
    lh = 1 - likelihood
    lh_vector = np.asarray(lh).flatten()

    lh_cubic = gd.CubicalComplex(
        #dimensions=[lh.shape[0], lh.shape[1]],
        dimensions=lh.shape, # allows 3D
        top_dimensional_cells=lh_vector
    )

    Diag_lh = lh_cubic.persistence(homology_coeff_field=2, min_persistence=0)
    pairs_lh = lh_cubic.cofaces_of_persistence_pairs()

    # If the paris is 0, return False to skip
    if (len(pairs_lh[0])==0): return 0, 0, 0, False
    if (len(pairs_lh[0][0])==0): return 0, 0, 0, False

    # return persistence diagram, birth/death critical points
    # Miguel: for some reason, the author only looks at one dimension
    # Not sure but probably you should read pairs_lh[0][0-1-2] (3 dims)
    pd_lh = np.array([[lh_vector[pairs_lh[0][0][i][0]], lh_vector[pairs_lh[0][0][i][1]]] for i in range(len(pairs_lh[0][0]))])
    # Make it 3D compatible
    #bcp_lh = np.array([[pairs_lh[0][0][i][0]//lh.shape[1], pairs_lh[0][0][i][0]%lh.shape[1]] for i in range(len(pairs_lh[0][0]))])
    #dcp_lh = np.array([[pairs_lh[0][0][i][1]//lh.shape[1], pairs_lh[0][0][i][1]%lh.shape[1]] for i in range(len(pairs_lh[0][0]))])
    bcp_lh = np.array(np.unravel_index(pairs_lh[0][0][:, 0], likelihood.shape)).T
    dcp_lh = np.array(np.unravel_index(pairs_lh[0][0][:, 1], likelihood.shape)).T

    if len(pd_lh.shape) != 2 or pd_lh.shape[1] != 2:
        print("pd_lh.shape", pd_lh.shape)
        from IPython import embed; embed(); asd

    # This is because I get some error
    #lh_pers = abs(lh_dgm[:, 1] - lh_dgm[:, 0])
    #IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
    #return pd_lh, bcp_lh, dcp_lh, (len(pd_lh.shape)==2 and pd_lh.shape[1] == 2)
    return pd_lh, bcp_lh, dcp_lh, True

