from __future__ import annotations

import numpy as np
import torch, warnings
from torch.nn.modules.loss import _Loss
from monai.networks import one_hot
from typing import Union
from scipy.ndimage import distance_transform_edt as dist
import scipy.ndimage as ndimage
import malis
from skimage import measure
import time

# Helpers, to avoid stuffing this file
import lib.losses_cldice as utils_cldice
import lib.losses_warping as utils_warping
import lib.losses_topoloss as utils_topoloss
import lib.losses_skeletonrecallloss as utils_srl
import lib.losses_bettimatching as utils_bettimatching
import lib.losses_cbdice as utils_cbdice

def _dice_loss(input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    axis = list([i for i in range(2, len(target.shape))])
    num = 2 * torch.sum(input * target, axis=axis) + 1e-5
    denom = torch.clip( torch.sum(target, axis=axis) + torch.sum(input, axis=axis) + 1e-5, 1e-8)
    dice_loss = (num / denom) * weights
    return -torch.mean(dice_loss)

def _crossentropy_loss(input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    #ce = torch.sum(target * torch.log(input + 1e-15), axis=1)
    #ce_loss = -torch.mean(ce)
    axis = list(range(len(input.shape)))
    axis.remove(1)
    ce = -torch.mean(target * torch.log(input + 1e-15), axis=axis)
    ce_loss = torch.sum(ce *  weights)
    return ce_loss

class RegionWiseLoss(_Loss):
    """
    """

    def __init__(self, rwmap_type:str="rrwmap") -> None:
        super().__init__()
        self.to_onehot = True
        self.ratio = [1, 1]
        if rwmap_type == "rrwmap":
            self.compute_rwmap = self._compute_rrwmap
        elif rwmap_type == "rrwmap_selfdistillation":
            self.compute_rwmap = self._compute_rrwmap_selfdistillation
        else:
            raise ValueError(f"Unknown rwmap: `{rwmap_type}`")

    def _compute_rrwmap(self, y_true: np.array):
        rrwmap = np.zeros_like(y_true) # Y: one-hot encoded ground truth
        for b in range(rrwmap.shape[0]): # Batch dim
            for c in range(rrwmap.shape[1]): # Channel dim
                rrwmap[b, c] = dist(y_true[b, c])
                rrwmap[b, c] = -1 * (rrwmap[b, c] / (np.max(rrwmap[b, c] + 1e-15)))
                rrwmap[b, rrwmap[b]==0] = self.ratio[c]
        #rrwmap[rrwmap==0] = 1
        return rrwmap

    def _compute_rrwmap_selfdistillation(self, y_true: np.array):

        rrwmap = torch.zeros_like(y_true) # Y: one-hot encoded ground truth
        #y_true_cpu = y_true.cpu().detach().numpy()
        for b in range(rrwmap.shape[0]): # Batch dim
            rrwmap[b] = y_true[b] * -1
            rrwmap[b, 0][y_true[b, 1] > 0.9] = 1
            rrwmap[b, 1][y_true[b, 1] < 0.1] = 1
        return rrwmap

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        """
        # CREMI and selfdistillation
        input = torch.softmax(input, 1)
        if self.to_onehot:
            target = one_hot(target, num_classes=input.shape[1])
            rwmap = torch.tensor(self._compute_rrwmap(target), device=input.device)
        else:
            if target.shape[1] == 1:
                target = one_hot(target, num_classes=input.shape[1])
                rwmap = torch.tensor(self._compute_rrwmap_selfdistillation(target), device=input.device)
            else:
                rwmap = torch.tensor(self._compute_rrwmap(target), device=input.device)

        loss = torch.mean(input * rwmap)
        return loss

    def forward_original(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        """
        input = torch.softmax(input, 1)
        if self.to_onehot:
            target = one_hot(target, num_classes=input.shape[1])

        rwmap = torch.tensor(self.compute_rwmap(target), device=input.device)
        loss = torch.mean(input * rwmap)
        return loss

class DiceLoss(_Loss):
    """Dice loss.
    """
    def __init__(self):
        super().__init__()
        self.to_onehot = True

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        input = torch.softmax(input, 1)
        if self.to_onehot:
            target = one_hot(target, num_classes=input.shape[1])

        dice_loss = _dice_loss(input, target)
        return dice_loss

class CrossEntropyLoss(_Loss):
    """Dice loss.
    """
    def __init__(self):
        super().__init__()
        self.to_onehot = True

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        input = torch.softmax(input, 1)
        if self.to_onehot:
            target = one_hot(target, num_classes=input.shape[1])

        ce_loss = _crossentropy_loss(input, target)
        return ce_loss


class CEDiceLoss(_Loss):
    """
    """
    def __init__(self, lambda_ce: float=1, lambda_dice: float=1,
            weights = 1) -> None:
        super().__init__()
        self.dice_loss_fn = _dice_loss
        self.ce_loss_fn = _crossentropy_loss
        self.lambda_ce = lambda_ce
        self.lambda_dice = lambda_dice
        self.to_onehot = True
        if isinstance(weights, (list, tuple)):
            self.weights = torch.Tensor([weights]).cuda()
        elif isinstance(weights, int): # int
            self.weights = torch.Tensor([[weights]]).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        input = torch.softmax(input, 1)
        if self.to_onehot:
            target = one_hot(target, num_classes=input.shape[1])

        # Dice loss
        dice_loss = 0
        if self.lambda_dice > 0:
            dice_loss = self.dice_loss_fn(input, target, self.weights)

        # Cross Entropy loss
        ce_loss = 0
        if self.lambda_ce > 0:
            ce_loss = self.ce_loss_fn(input, target, self.weights)

        loss = dice_loss*self.lambda_dice + ce_loss*self.lambda_ce
        return loss

class clDiceLoss(_Loss):
    def __init__(self, iterations=3, alpha=0.5, smooth = 1., weights = 1) -> None:
        super().__init__()
        self.its = iterations # This parameter increases drammatically GPU mem.
        self.smooth = smooth
        self.alpha = alpha
        self.to_onehot = True
        self.dice_loss_fn = _dice_loss
        if isinstance(weights, (list, tuple)):
            self.weights = torch.Tensor([weights]).cuda()
        elif isinstance(weights, int): # int
            self.weights = torch.Tensor([[weights]]).cuda()

    #def forward(self, y_pred, y_true):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        input = torch.softmax(input, 1)
        if self.to_onehot:
            target = one_hot(target, num_classes=input.shape[1])

        # Dice loss
        dice_loss = self.dice_loss_fn(input, target, self.weights)

        # cl Dice
        skel_pred = utils_cldice.soft_skel(input, self.its)
        skel_true = utils_cldice.soft_skel(target, self.its)

        tprec = (torch.sum((skel_pred * target)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)
        tsens = (torch.sum((skel_true * input)[:,1:,...])+self.smooth)/(torch.sum(skel_true[:,1:,...])+self.smooth)
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)

        return (1.0-self.alpha)*dice_loss+self.alpha*cl_dice

class WarpingLoss(_Loss):
    """
    Calculate the warping loss of the predicted image and ground truth image
    Args:
        pre:   The likelihood pytorch tensor for neural networks.
        gt:   The groundtruth of pytorch tensor.
    Returns:
        warping_loss:   The warping loss value (tensor)
    """
    ## compute false positive and false negative
    def __init__(self, lambd: float=0.1, activate_when: float=0.7, weights=1) -> None:
        super().__init__()
        self.lambd = lambd
        self.activate = False
        self.activate_when = activate_when
        self.to_onehot = True
        self.dice_loss_fn = _dice_loss
        self.ce_loss_fn = _crossentropy_loss
        if isinstance(weights, (list, tuple)):
            self.weights = torch.Tensor([weights]).cuda()
        elif isinstance(weights, int): # int
            self.weights = torch.Tensor([[weights]]).cuda()

    #def forward(self, y_pred: torch.Tensor,
    #            y_gt: torch.Tensor) -> torch.Tensor:
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        if self.to_onehot:
            target = one_hot(target, num_classes=input.shape[1])
        input = torch.softmax(input, 1)

        # Partial dice
        dice_loss = self.dice_loss_fn(input, target, self.weights)
        if not self.activate:
            return dice_loss

        if (len(input.shape) == 4):
            B, C, H, W = input.shape
            critical_points = np.zeros((B,H,W))
        else:
            B, C, H, W, Z = input.shape
            critical_points = np.zeros((B,H,W,Z))

        pre = torch.argmax(input, dim=1)
        # NOTE: Only for background vs. foreground segmentation
        # Applying this to multiclass is probably very costly
        y_target = target[:, 1:2]
        target = target[:, 0] # These are the same, but the dims are diff.

        pre = pre.cpu().detach().numpy().astype('uint8')
        target = target.cpu().detach().numpy().astype('uint8')

        pre_copy = pre.copy() # BHW
        target_copy = target.copy() # BHW

        for i in range(B):
            false_positive = ((pre_copy[i] - target_copy[i]) == 1).astype(int)
            false_negative = ((target_copy[i] - pre_copy[i]) == 1).astype(int)

            ## Use distance transform to determine the flipping order
            false_negative_distance_target = ndimage.distance_transform_edt(target_copy[i]) * false_negative  # shrink target while keep connected
            false_positive_distance_target = ndimage.distance_transform_edt(1 - target_copy[i]) * false_positive  # grow target while keep unconnected

            target_warp = utils_warping.update_simple_point(false_negative_distance_target, target_copy[i])
            target_warp = utils_warping.update_simple_point(false_positive_distance_target, target_warp)

            false_positive_distance_pre = ndimage.distance_transform_edt(pre_copy[i]) * false_positive  # shrink pre while keep connected
            false_negative_distance_pre = ndimage.distance_transform_edt(1-pre_copy[i]) * false_negative # grow target while keep unconnected
            pre_warp = utils_warping.update_simple_point(false_positive_distance_pre, pre_copy[i])
            pre_warp = utils_warping.update_simple_point(false_negative_distance_pre, pre_warp)

            critical_points[i] = np.logical_or(np.not_equal(pre[i], target_warp), np.not_equal(target[i], pre_warp)).astype(int)

        warping_loss = self.ce_loss_fn(input * torch.unsqueeze(torch.from_numpy(critical_points), dim=1).cuda(), one_hot(y_target * torch.unsqueeze(torch.from_numpy(critical_points), dim=1).cuda(), num_classes=input.shape[1]), 1)

        loss = dice_loss + self.lambd*warping_loss
        return loss


class TopoLoss(_Loss):
    """
    Calculate the topoloss loss of the predicted image and ground truth image
    Args:
        pre:   The likelihood pytorch tensor for neural networks.
        gt:   The groundtruth of pytorch tensor.
    Returns:
        warping_loss:   The warping loss value (tensor)
    """
    ## compute false positive and false negative
    def __init__(self, lambd: float=100.0, topo_size: int=50,
                 activate_when: float=0.7, weights=1):
        super().__init__()
        self.lambd = lambd
        self.topo_size = topo_size
        self.activate = False # Change this
        self.activate_when = activate_when
        self.to_onehot = True
        self.ce_loss_fn = _crossentropy_loss
        if isinstance(weights, (list, tuple)):
            self.weights = torch.Tensor([weights]).cuda()
        elif isinstance(weights, int): # int
            self.weights = torch.Tensor([[weights]]).cuda()

    #def forward(self, y_pred: torch.Tensor,
    #            y_gt: torch.Tensor) -> torch.Tensor:
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        if self.to_onehot:
            target = one_hot(target, num_classes=input.shape[1])
        input = torch.softmax(input, 1)

        #self.c += 1
        celoss = self.ce_loss_fn(input, target, self.weights)
        if not self.activate:
            return celoss

        if len(input.shape) == 5:
            topoloss = self._topoloss_3D(input, target)
        else:
            topoloss = self._topoloss_2D(input, target)

        return celoss + self.lambd * topoloss

    def _topoloss_2D(self, input, target):

        # NOTE THAT THIS IS NOT MULTI-CLASS
        input = torch.softmax(input, 1)[:, 1]
        likelihood = input.clone().cpu().detach().numpy()
        gt = target[:, 0].clone().cpu().detach().numpy()
        # likelihood \in [0,1]
        # gt \in {0,1}

        topo_cp_weight_map = np.zeros(likelihood.shape)
        topo_cp_ref_map = np.zeros(likelihood.shape)

        idxs = [(b, y, x) for b in range(likelihood.shape[0]) for y in range(0, likelihood.shape[1], self.topo_size) for x in range(0, likelihood.shape[2], self.topo_size)]

        #for y in range(0, likelihood.shape[0], self.topo_size):
        #    for x in range(0, likelihood.shape[1], self.topo_size):
        #        for z in range(0, likelihood.shape[2], self.topo_size):

        for (b, y, x) in idxs:

            # 3D compatible
            lh_patch = likelihood[b, y:min(y + self.topo_size, likelihood.shape[1]),
                         x:min(x + self.topo_size, likelihood.shape[2])]
            gt_patch = gt[b, y:min(y + self.topo_size, gt.shape[1]),
                         x:min(x + self.topo_size, gt.shape[2])]

            if(np.min(lh_patch) == 1 or np.max(lh_patch) == 0): continue
            if(np.min(gt_patch) == 1 or np.max(gt_patch) == 0): continue

            # Get the critical points of predictions and ground truth
            pd_lh, bcp_lh, dcp_lh, pairs_lh_pa = utils_topoloss.getCriticalPoints(lh_patch)
            pd_gt, bcp_gt, dcp_gt, pairs_lh_gt = utils_topoloss.getCriticalPoints(gt_patch)

            # If the pairs not exist, continue for the next loop
            if not(pairs_lh_pa): continue
            if not(pairs_lh_gt): continue

            # Miguel: "force_list" is not used at all
            force_list, idx_holes_to_fix, idx_holes_to_remove = utils_topoloss.compute_dgm_force(pd_lh, pd_gt, pers_thd=0.03)

            #from IPython import embed; embed(); asd

            # If there are holes to fix or remove, for each hole, check
            # the indx is within the allowed coordinates (>0 & <shape)
            if (len(idx_holes_to_fix) > 0 or len(idx_holes_to_remove) > 0):
                for hole_indx in idx_holes_to_fix:
                    #from IPython import embed; embed(); asd
                    if ((bcp_lh[hole_indx] >= 0) & (bcp_lh[hole_indx] < likelihood.shape[1:])).all():
                        iy = y + int(bcp_lh[hole_indx][0])
                        ix = x + int(bcp_lh[hole_indx][1])
                        # push birth to 0 i.e. min birth prob or likelihood
                        topo_cp_weight_map[b, iy, ix] = 1
                        topo_cp_ref_map[b, iy, ix] = 0

                    if ((dcp_lh[hole_indx] >= 0) & (dcp_lh[hole_indx] < likelihood.shape[1:])).all():
                        iy = y + int(dcp_lh[hole_indx][0])
                        ix = x + int(dcp_lh[hole_indx][1])

                        # push death to 1 i.e. max death prob or likelihood
                        topo_cp_weight_map[b, iy, ix] = 1
                        topo_cp_ref_map[b, iy, ix] = 1

                for hole_indx in idx_holes_to_remove:
                    if ((bcp_lh[hole_indx] >= 0) & (bcp_lh[hole_indx] < likelihood.shape[1:])).all():

                        iy = y + int(bcp_lh[hole_indx][0])
                        ix = x + int(bcp_lh[hole_indx][1])
                        # push birth to death  # push to diagonal
                        topo_cp_weight_map[b, iy, ix] = 1

                        if ((dcp_lh[hole_indx] >= 0) & (dcp_lh[hole_indx] < likelihood.shape[1:])).all():
                            iiy = int(dcp_lh[hole_indx][0])
                            iix = int(dcp_lh[hole_indx][1])

                            topo_cp_ref_map[b, iy, ix] = lh_patch[iiy, iix]
                        else:
                            topo_cp_ref_map[b, iy, ix] = 1

                    if ((dcp_lh[hole_indx] >= 0) & (dcp_lh[hole_indx] < likelihood.shape[1:])).all():
                        iy = int(dcp_lh[hole_indx][0])
                        ix = int(dcp_lh[hole_indx][1])

                        # push death to birth # push to diagonal
                        topo_cp_weight_map[b, iy, ix] = 1

                        if ((bcp_lh[hole_indx] >= 0) & (bcp_lh[hole_indx] < likelihood.shape[1:])).all():
                            iiy = int(bcp_lh[hole_indx][0])
                            iix = int(bcp_lh[hole_indx][1])

                            topo_cp_ref_map[b, iy, ix] = lh_patch[iiy, iix]
                        else:
                            topo_cp_ref_map[b, iy, ix] = 0

        ####
        topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float).cuda()
        topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float).cuda()

        # Measuring the MSE loss between predicted critical points and reference critical points
        loss_topo = (((input * topo_cp_weight_map) - topo_cp_ref_map) ** 2).mean()
        return loss_topo

    def _topoloss_3D(self, input, target):

        # NOTE THAT THIS IS NOT MULTI-CLASS
        input = torch.softmax(input, 1)[:, 1]
        likelihood = input.clone().cpu().detach().numpy()
        gt = target[:, 0].clone().cpu().detach().numpy()
        # likelihood \in [0,1]
        # gt \in {0,1}

        topo_cp_weight_map = np.zeros(likelihood.shape)
        topo_cp_ref_map = np.zeros(likelihood.shape)

        idxs = [(b, y, x, z) for b in range(likelihood.shape[0]) for y in range(0, likelihood.shape[1], self.topo_size) for x in range(0, likelihood.shape[2], self.topo_size) for z in range(0, likelihood.shape[3], self.topo_size)]

        #for y in range(0, likelihood.shape[0], self.topo_size):
        #    for x in range(0, likelihood.shape[1], self.topo_size):
        #        for z in range(0, likelihood.shape[2], self.topo_size):

        for (b, y, x, z) in idxs:

            # 3D compatible
            lh_patch = likelihood[b, y:min(y + self.topo_size, likelihood.shape[1]),
                         x:min(x + self.topo_size, likelihood.shape[2]),
                         z:min(z + self.topo_size, likelihood.shape[3])]
            gt_patch = gt[b, y:min(y + self.topo_size, gt.shape[1]),
                         x:min(x + self.topo_size, gt.shape[2]),
                         z:min(z + self.topo_size, gt.shape[3])]

            if(np.min(lh_patch) == 1 or np.max(lh_patch) == 0): continue
            if(np.min(gt_patch) == 1 or np.max(gt_patch) == 0): continue

            # Get the critical points of predictions and ground truth
            pd_lh, bcp_lh, dcp_lh, pairs_lh_pa = utils_topoloss.getCriticalPoints(lh_patch)
            pd_gt, bcp_gt, dcp_gt, pairs_lh_gt = utils_topoloss.getCriticalPoints(gt_patch)

            # If the pairs not exist, continue for the next loop
            if not(pairs_lh_pa): continue
            if not(pairs_lh_gt): continue

            # Miguel: "force_list" is not used at all
            force_list, idx_holes_to_fix, idx_holes_to_remove = utils_topoloss.compute_dgm_force(pd_lh, pd_gt, pers_thd=0.03)

            #from IPython import embed; embed(); asd

            # If there are holes to fix or remove, for each hole, check
            # the indx is within the allowed coordinates (>0 & <shape)
            if (len(idx_holes_to_fix) > 0 or len(idx_holes_to_remove) > 0):
                for hole_indx in idx_holes_to_fix:
                    #from IPython import embed; embed(); asd
                    if ((bcp_lh[hole_indx] >= 0) & (bcp_lh[hole_indx] < likelihood.shape[1:])).all():
                        iy = y + int(bcp_lh[hole_indx][0])
                        ix = x + int(bcp_lh[hole_indx][1])
                        iz = z + int(bcp_lh[hole_indx][2])
                        # push birth to 0 i.e. min birth prob or likelihood
                        topo_cp_weight_map[b, iy, ix, iz] = 1
                        topo_cp_ref_map[b, iy, ix, iz] = 0

                    if ((dcp_lh[hole_indx] >= 0) & (dcp_lh[hole_indx] < likelihood.shape[1:])).all():
                        iy = y + int(dcp_lh[hole_indx][0])
                        ix = x + int(dcp_lh[hole_indx][1])
                        iz = z + int(dcp_lh[hole_indx][2])

                        # push death to 1 i.e. max death prob or likelihood
                        topo_cp_weight_map[b, iy, ix, iz] = 1
                        topo_cp_ref_map[b, iy, ix, iz] = 1

                for hole_indx in idx_holes_to_remove:
                    if ((bcp_lh[hole_indx] >= 0) & (bcp_lh[hole_indx] < likelihood.shape[1:])).all():

                        iy = y + int(bcp_lh[hole_indx][0])
                        ix = x + int(bcp_lh[hole_indx][1])
                        iz = z + int(bcp_lh[hole_indx][2])
                        # push birth to death  # push to diagonal
                        topo_cp_weight_map[b, iy, ix, iz] = 1

                        if ((dcp_lh[hole_indx] >= 0) & (dcp_lh[hole_indx] < likelihood.shape[1:])).all():
                            iiy = int(dcp_lh[hole_indx][0])
                            iix = int(dcp_lh[hole_indx][1])
                            iiz = int(dcp_lh[hole_indx][2])

                            topo_cp_ref_map[b, iy, ix, iz] = lh_patch[iiy, iix, iiz]
                        else:
                            topo_cp_ref_map[b, iy, ix, iz] = 1

                    if ((dcp_lh[hole_indx] >= 0) & (dcp_lh[hole_indx] < likelihood.shape[1:])).all():
                        iy = int(dcp_lh[hole_indx][0])
                        ix = int(dcp_lh[hole_indx][1])
                        iz = int(dcp_lh[hole_indx][2])

                        # push death to birth # push to diagonal
                        topo_cp_weight_map[b, iy, ix, iz] = 1

                        if ((bcp_lh[hole_indx] >= 0) & (bcp_lh[hole_indx] < likelihood.shape[1:])).all():
                            iiy = int(bcp_lh[hole_indx][0])
                            iix = int(bcp_lh[hole_indx][1])
                            iiz = int(bcp_lh[hole_indx][2])

                            topo_cp_ref_map[b, iy, ix, iz] = lh_patch[iiy, iix, iiz]
                        else:
                            topo_cp_ref_map[b, iy, ix, iz] = 0

        ####
        topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float).cuda()
        topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float).cuda()

        # Measuring the MSE loss between predicted critical points and reference critical points
        loss_topo = (((input * topo_cp_weight_map) - topo_cp_ref_map) ** 2).mean()
        return loss_topo

class SkeletonRecallLoss(_Loss):
    def __init__(self):
        super().__init__()
        self.apply_softmax = True
        self.smooth = 1e-5
        self.to_onehot = True

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        shp_inp, shp_tar = input.shape, target.shape
        if not self.to_onehot:
            # This will happen when self-training, when target is a soft label.
            target = torch.argmax(target, dim=1).unsqueeze(1)

        skeletons = utils_srl.compute_skeletons(target).to(target.device)
        target = one_hot(target, num_classes=input.shape[1])

        y_onehot = target[:, 1:]

        if self.apply_softmax:
            input = torch.softmax(input, 1)

        input = input[:, 1:]
        axes = list(range(2, len(shp_inp)))

        with torch.no_grad():
            #sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)
            sum_gt = (y_onehot * skeletons).sum(axes)

        #inter_rec = (input * y_onehot).sum(axes) if loss_mask is None else (x * y_onehot * loss_mask).sum(axes)
        inter_rec = (input * y_onehot * skeletons).sum(axes)
        rec = (inter_rec + self.smooth) / (torch.clip(sum_gt+self.smooth, 1e-8))
        rec = rec.mean()
        return -rec

class CESkeletonRecallLoss(_Loss):
    def __init__(self, lambd=1, weights=1):
        super().__init__()
        self.w = lambd
        self.ce_loss_fn = _crossentropy_loss
        self.srl_fun = SkeletonRecallLoss()
        self.srl_fun.apply_softmax = False
        self.to_onehot = True

        if isinstance(weights, (list, tuple)):
            self.weights = torch.Tensor([weights]).cuda()
        elif isinstance(weights, int): # int
            self.weights = torch.Tensor([[weights]]).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        #if self.to_onehot:
        #    target = one_hot(target, num_classes=input.shape[1])
        input = torch.softmax(input, 1)

        if self.to_onehot:
            celoss = self.ce_loss_fn(input,
                            one_hot(target, num_classes=input.shape[1]),
                            self.weights)
            self.srl_fun.to_onehot = True
        else:
            celoss = self.ce_loss_fn(input, target, self.weights)
            self.srl_fun.to_onehot = False

        srl = self.srl_fun(input, target)

        return celoss + self.w * srl


class cbDiceLoss(_Loss):
    #def __init__(self, iter_=10, smooth = 1., ):
    def __init__(self,smooth = 1., alpha = 1, beta = 1, weights = 1):
        super().__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.to_onehot = True
        self.dice_loss_fn = _dice_loss
        self.ce_loss_fn = _crossentropy_loss
        if isinstance(weights, (list, tuple)):
            self.weights = torch.Tensor([weights]).cuda()
        elif isinstance(weights, int): # int
            self.weights = torch.Tensor([[weights]]).cuda()

        # Topology-preserving skeletonization: https://github.com/martinmenten/skeletonization-for-gradient-based-optimization
        # This is the one used in the paper
        self.t_skeletonize = utils_cbdice.Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic')

        # Morphological skeletonization: https://github.com/jocpae/clDice/tree/master/cldice_loss/pytorch
        #self.m_skeletonize = SoftSkeletonize(num_iter=iter_)

    #def forward(self, y_pred, y_true):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if len(target.shape) == 4:
            dim = 2
        elif len(target.shape) == 5:
            dim = 3
        else:
            raise ValueError("target should be 4D or 5D tensor.")

        if self.to_onehot:
            target = one_hot(target, num_classes=input.shape[1])
        # Note, at this point, target.shape = B,C,H,W

        # Dice loss
        dice_loss = self.dice_loss_fn(torch.softmax(input, 1), target, self.weights)

        # Cross Entropy loss
        ce_loss = self.ce_loss_fn(torch.softmax(input, 1), target, self.weights)


        y_pred_fore = input[:, 1:]
        y_pred_fore = torch.max(y_pred_fore, dim=1, keepdim=True)[0] # C foreground channels -> 1 channel
        y_pred_binary = torch.cat([input[:, :1], y_pred_fore], dim=1)
        y_prob_binary = torch.softmax(y_pred_binary, 1)
        y_pred_prob = y_prob_binary[:, 1] # predicted probability map of foreground

        with torch.no_grad():
            # Put target.shape back to B,1,H,W
            # This step is important for self-training
            target = torch.argmax(target, 1, keepdim=True)
            target = torch.where(target > 0, 1, 0).squeeze(1).float() # ground truth of foreground
            y_pred_hard = (y_pred_prob > 0.5).float()

            #if t_skeletonize_flage:
            skel_pred_hard = self.t_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
            skel_true = self.t_skeletonize(target.unsqueeze(1)).squeeze(1)

            #else:
            #    skel_pred_hard = self.m_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
            #    skel_true = self.m_skeletonize(target.unsqueeze(1)).squeeze(1)

        skel_pred_prob = skel_pred_hard * y_pred_prob

        q_vl, q_slvl, q_sl = utils_cbdice.get_weights(target, skel_true, dim, prob_flag=False)
        q_vp, q_spvp, q_sp = utils_cbdice.get_weights(y_pred_prob, skel_pred_prob, dim, prob_flag=True)

        w_tprec = (torch.sum(torch.multiply(q_sp, q_vl))+self.smooth)/(torch.sum(utils_cbdice.combine_tensors(q_spvp, q_slvl, q_sp))+self.smooth)
        w_tsens = (torch.sum(torch.multiply(q_sl, q_vp))+self.smooth)/(torch.sum(utils_cbdice.combine_tensors(q_slvl, q_spvp, q_sl))+self.smooth)

        cb_dice_loss = - 2.0 * (w_tprec * w_tsens) / (w_tprec + w_tsens)

        # Eq. in Sec 3.2
        w1 = 0.5
        w2 = (self.alpha/(2*(self.alpha+self.beta)))
        w3 = (self.beta/(2*(self.alpha+self.beta)))

        loss = w1*ce_loss + w2*dice_loss + w3*cb_dice_loss

        return loss

class MSETopoWinLoss(_Loss):

    # This loss function will only work with two classes, background and foreground
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9409952

    def __init__(self, weights=1):
        super().__init__()
        self.malis_lr = 1
        self.alpha = 1e-3
        self.malis_lr_pos = 0.1 # Beta
        self.window = 64
        self.Dmax = 20
        # I'm not sure where this comes in the paper. It might be that it's
        # 5*2=10 (5=dilation)
        self.Dmin = 10

        if isinstance(weights, int): # int
            self.weights = [weights, weights]
        else:
            self.weights = weights

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Unclear but it makes sense
        input = torch.relu(input)

        pred_np_full = input.cpu().detach().numpy()
        target_np_full = target.cpu().detach().numpy()
        B,C,H,W = pred_np_full.shape

        mse_loss = 0
        for b in range(B):
            dist_map = np.clip(dist(target_np_full[b, 0]), a_min=None, a_max=self.Dmax)
            tmp = ((input[b, 0] - torch.from_numpy(dist_map).cuda())**2)
            # For class imbalance
            mse_loss += ( (target[b, 0]*self.weights[1]*tmp) + ((1-target[b,0])*self.weights[0]*tmp) ).sum()
        #topo_loss = (target*self.weights[0]*topo_loss) + ((1-target)*self.weights[1]*topo_loss)

        weights_n = np.zeros(pred_np_full.shape)
        weights_p = np.zeros(pred_np_full.shape)
        w = self.window

        for k in range(H // w):
            for j in range(W // w):
                pred_np = pred_np_full[:,:,k*w:(k+1)*w,j*w:(j+1)*w]
                target_np = target_np_full[:,:,k*w:(k+1)*w,j*w:(j+1)*w]

                nodes_indexes = np.arange(w*w).reshape(w,w)
                nodes_indexes_h = np.vstack([nodes_indexes[:,:-1].ravel(), nodes_indexes[:,1:].ravel()]).tolist()
                nodes_indexes_v = np.vstack([nodes_indexes[:-1,:].ravel(), nodes_indexes[1:,:].ravel()]).tolist()
                nodes_indexes = np.hstack([nodes_indexes_h, nodes_indexes_v])
                nodes_indexes = np.uint64(nodes_indexes)

                costs_h = (pred_np[:,:,:,:-1] + pred_np[:,:,:,1:]).reshape(B,-1)
                costs_v = (pred_np[:,:,:-1,:] + pred_np[:,:,1:,:]).reshape(B,-1)
                costs = np.hstack([costs_h, costs_v])
                costs = np.float32(costs)

                gtcosts_h = (target_np[:,:,:,:-1] + target_np[:,:,:,1:]).reshape(B,-1)
                gtcosts_v = (target_np[:,:,:-1,:] + target_np[:,:,1:,:]).reshape(B,-1)
                gtcosts = np.hstack([gtcosts_h, gtcosts_v])
                gtcosts = np.float32(gtcosts)

                costs_n = costs.copy()
                costs_p = costs.copy()

                costs_n[gtcosts > self.Dmax] = self.Dmax
                costs_p[gtcosts < self.Dmin] = 0
                gtcosts[gtcosts > self.Dmax] = self.Dmax

                for i in range(len(pred_np)):
                    sg_gt = ndimage.label(ndimage.binary_dilation((target_np[i,0] == 0), iterations=5)==0)[0]

                    edge_weights_n = malis.malis_loss_weights(sg_gt.astype(np.uint64).flatten(), nodes_indexes[0], \
                                           nodes_indexes[1], costs_n[i], 0)

                    edge_weights_p = malis.malis_loss_weights(sg_gt.astype(np.uint64).flatten(), nodes_indexes[0], \
                                           nodes_indexes[1], costs_p[i], 1)


                    num_pairs_n = np.sum(edge_weights_n)
                    if num_pairs_n > 0:
                        edge_weights_n = edge_weights_n/num_pairs_n

                    num_pairs_p = np.sum(edge_weights_p)
                    if num_pairs_p > 0:
                        edge_weights_p = edge_weights_p/num_pairs_p

                    ## Depending on your clip values
                    edge_weights_n[gtcosts[i] >= self.Dmin] = 0
                    edge_weights_p[gtcosts[i] < self.Dmax] = 0

                    malis_w = edge_weights_n.copy()

                    malis_w_h, malis_w_v = np.split(malis_w, 2)
                    malis_w_h, malis_w_v = malis_w_h.reshape(w,w-1), malis_w_v.reshape(w-1,w)

                    nodes_weights = np.zeros((w,w), np.float32)
                    nodes_weights[:,:-1] += malis_w_h
                    nodes_weights[:,1:] += malis_w_h
                    nodes_weights[:-1,:] += malis_w_v
                    nodes_weights[1:,:] += malis_w_v

                    weights_n[i, 0, k*w:(k+1)*w, j*w:(j+1)*w] = nodes_weights

                    malis_w = edge_weights_p.copy()

                    malis_w_h, malis_w_v = np.split(malis_w, 2)
                    malis_w_h, malis_w_v = malis_w_h.reshape(w,w-1), malis_w_v.reshape(w-1,w)

                    nodes_weights = np.zeros((w,w), np.float32)
                    nodes_weights[:,:-1] += malis_w_h
                    nodes_weights[:,1:] += malis_w_h
                    nodes_weights[:-1,:] += malis_w_v
                    nodes_weights[1:,:] += malis_w_v

                    weights_p[i, 0, k*w:(k+1)*w, j*w:(j+1)*w] = nodes_weights

        loss_n = (input).pow(2)

        loss_p = (self.Dmax - input).pow(2)
        topo_loss = self.malis_lr * loss_n * torch.Tensor(weights_n).cuda() + self.malis_lr_pos * loss_p * torch.Tensor(weights_p).cuda()

        # For class imbalance
        topo_loss = (target*self.weights[1]*topo_loss) + ((1-target)*self.weights[0]*topo_loss)

        loss = self.alpha*mse_loss + topo_loss.sum()

        return loss



#####################################
# LOSS FUNCTIONS I COULDN'T UTILIZE #
#####################################


class DiceBettiMatchingLoss(_Loss):

    def __init__(
        self,
        batch: bool = False,
        alpha: float = 0.5,
        relative=True,
        activate_when: float = 0.0,
        filtration='superlevel',
        weights=1
    ) -> None:
        super().__init__()
        #msg = "This loss function was not ultimately adapted to our framework. Read notes"
        msg = ("This loss function was not ultimately adapted to our framework due to\n"
        "its extremely long running time.\n\n"
        "In our preliminary experiments, running BettiMatchingLoss on a single\n"
        "prediction took more than 10 minutes (we don't know exactly how long\n"
        "because we interrupted the experiment after noticing its running time).\n\n"
        "In the best scenario, considering that execution time took 10 minutes,\n"
        "our experimental setup of 12000 iterations on batches of 10 images would\n"
        "take approximately 2.3 years per experiment: (12000*10*10)/(365*24*60).\n"
        "Since we wanted to keep the experimental setting fixed for a fair comparison,\n"
        "we were unable to use BettiMatching loss.\n\n"
        "Note that the experiments in the original paper were conducted on\n"
        "patches of 48x48 (See Appendix K of the BettiMatching paper), and\n"
        "BettiMatchingLoss is O(n^3) and reportely slow in practice (Appendix E).\n\n"
        "Note 2: See the Appendices in the published paper. Appendix E doesn't appear\n"
        "in the arxiv version: https://proceedings.mlr.press/v202/stucki23a/stucki23a.pdf\n")

        #raise Exception(msg)
        self.batch = batch
        self.alpha = alpha
        self.relative = relative
        self.filtration = filtration
        self.activate = False # Change this
        self.activate_when = activate_when
        self.window = 48 # As in the paper
        self.num_patches = 8 # As in the paper
        # Added
        self.dice_loss_fn = _dice_loss
        self.to_onehot = True
        if isinstance(weights, (list, tuple)):
            self.weights = torch.Tensor([weights]).cuda()
        elif isinstance(weights, int): # int
            self.weights = torch.Tensor([[weights]]).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = torch.softmax(input, 1)


        #t0 = time.time(); t = utils_bettimatching.compute_BettiMatchingLoss(pair, sigmoid=False, filtration=self.filtration, relative=self.relative); t1 = time.time()

        if self.activate:
            if self.to_onehot:
                target2 = target
            else:
                target2 = torch.argmax(target, 1, keepdim=True)

            for b in range(target2.shape[0]):
                idx_y = np.random.randint(0, target2.shape[2]-self.window, self.num_patches)
                idx_x = np.random.randint(0, target2.shape[3]-self.window, self.num_patches)
                i = 0
                pair = (input[b,:,idx_y[i]:idx_y[i]+self.window, idx_x[i]:idx_x[i]+self.window],
                        target2[b,:,idx_y[i]:idx_y[i]+self.window, idx_x[i]:idx_x[i]+self.window] )
                betti_loss = utils_bettimatching.compute_BettiMatchingLoss(pair, sigmoid=False, filtration=self.filtration, relative=self.relative).cuda()
                for i in range(1, self.num_patches):
                    pair = (input[b,:,idx_y[i]:idx_y[i]+self.window, idx_x[i]:idx_x[i]+self.window],
                            target2[b,:,idx_y[i]:idx_y[i]+self.window, idx_x[i]:idx_x[i]+self.window] )
                    betti_loss += utils_bettimatching.compute_BettiMatchingLoss(pair, sigmoid=False, filtration=self.filtration, relative=self.relative).cuda()

        if self.to_onehot:
            target = one_hot(target, num_classes=input.shape[1])
        dice_loss = self.dice_loss_fn(input, target, self.weights)

        if self.activate:
            denom = target.shape[0] *  self.num_patches # Eq. to do "mean"
            loss = dice_loss + self.alpha * betti_loss / denom
        else:
            loss = dice_loss

        return loss




class TopoGradLoss(_Loss):
    #def __init__(self, iter_=10, smooth = 1., ):
    def __init__(self, activate_when: float=0.7, weights=1) -> None:
        super().__init__()
        raise Exception("stop")
        # I couldn't continue because I'm unable to install topologylayer library
        # https://github.com/bruel-gabrielsson/TopologyLayer/issues/49

        self.activate = False
        self.activate_when = activate_when
        self.to_onehot = True
        self.ce_loss_fn = _crossentropy_loss

        if isinstance(weights, (list, tuple)):
            self.weights = torch.Tensor([weights]).cuda()
        elif isinstance(weights, int): # int
            self.weights = torch.Tensor([[weights]]).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        if self.to_onehot:
            target = one_hot(target, num_classes=input.shape[1])
        input = torch.softmax(input, 1)

        celoss = self.ce_loss_fn(input, target, self.weights)
        if not self.activate:
            return celoss


        Z_cpu = Z_cuda.cpu()
        a = dgminfo(Z_cpu)


        L0 = (TopKBarcodeLengths(dim=0, k=max_k)(a)**2).sum()
        dim_1_sq_bars = TopKBarcodeLengths(dim=1, k=max_k)(a)**2
        bar_signs = torch.ones(max_k)
        bar_signs[:H_i[1]] = -1
        L1 = (dim_1_sq_bars * bar_signs).sum()

        L_sqdiff = l2_loss(original_model_output, Z_cpu) * L_sqdiff_weight
        L = L0 + L1 + L_sqdiff

class CavityLoss(_Loss):
    def __init__(self, alpha=1, weights=1) -> None:
        super().__init__()

        raise Exception("stop")
        # I couldn't install MDB... it requires compiling C++ code

        self.alpha = alpha
        self.to_onehot = True
        self.ce_loss_fn = _crossentropy_loss
        self.dice_loss_fn = _dice_loss

        if isinstance(weights, (list, tuple)):
            self.weights = torch.Tensor([weights]).cuda()
        elif isinstance(weights, int): # int
            self.weights = torch.Tensor([[weights]]).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        if self.to_onehot:
            target = one_hot(target, num_classes=input.shape[1])
        input = torch.softmax(input, 1)

        celoss = self.ce_loss_fn(input, target, self.weights)
        dice_loss = self.dice_loss_fn(input, target, self.weights)
        cavity_loss = 0

        # To avoid compiling, and because it's simple, I re-implemented this
        # Here, I use the nomenclature from Fig. 2
        from IPython import embed; embed(); asd
        pred_cpu = input.cpu().detach().numpy()
        target_cpu = target.cpu().detach().numpy()
        ub = np.argmax(pred_cpu, 1) # B,H,W
        gt = np.argmax(target_cpu, 1) # B,H,W
        gt_hat = ndimage.binary_dilation(gt)*1

        um = gt_hat * ub
        u_diff = gt - um
        u_diff = (u_diff > 0)*1
        u_cc = (u_diff + um) < 0.5

        u_k = measure.label(u_cc,  background=0, connectivity=1)
        res = watershed(ub, markers=u_k)

        loss = ce_loss + dice_loss + self.alpha * cavity_loss


