from __future__ import annotations
import torch, os, subprocess
from monai.metrics import CumulativeIterationMetric
from monai.metrics import SurfaceDistanceMetric as MONAI_SDM
from pathlib import Path
from monai.utils import MetricReduction
import numpy as np
import gudhi as gd
import time, random
import nibabel as nib
from skimage.morphology import skeletonize
from medpy import metric
from monai.networks import one_hot
import itertools
from scipy import ndimage

class HD95Metric(CumulativeIterationMetric):
    # In order to ignore certain voxels/pixels when computing HD95, those
    # pixels, in the ground truth, should have a value larger than the number
    # of classes. For instance, for binary classification (i.e., 0: background,
    # 1: foreground), the pixels to be ignored could have values of 2, 3, etc.

    def __init__(self, voxres) -> None:
        super().__init__()
        self._buffers = None
        self.voxres = voxres

    def _compute_tensor(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_pred.shape = 1,C,H,W(,D); y_pred \in {0,1}
        # It's 1 instead of B because this function is called for each image
        # y_true.shape = 1,1,H,W(,D)
        # Its channels is 1 because it's not one-hot encoded because sometimes
        # I want to know where is certain class to ignore it.


        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        hd95_foreground = []
        for b in range(y_pred.shape[0]):
            for c in range(1, y_pred.shape[1]):
                #from IPython import embed; embed(); asd
                if y_pred[b,c].sum() == 0:
                    hd95_foreground.append(9999999)
                else:
                    hd95_foreground.append( metric.hd95(y_pred[b,c], y_true[b,0]==c,
                                                    self.voxres) )
        return torch.tensor([hd95_foreground])

    def aggregate(self, reduction: MetricReduction | str | None = None
                       ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # This will be problematic with multiple classes, but it's easy to fix
        return torch.cat(self._buffers[0], dim=0)

class DiceMetric(CumulativeIterationMetric):
    # In order to ignore certain voxels/pixels when computing Dice, those
    # pixels, in the ground truth, should have a value larger than the number
    # of classes. For instance, for binary classification (i.e., 0: background,
    # 1: foreground), the pixels to be ignored could have values of 2, 3, etc.

    def __init__(self) -> None:
        super().__init__()
        self._buffers = None

    def _compute_tensor(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_pred.shape = 1,C,H,W(,D); y_pred \in {0,1}
        # It's 1 instead of B because this function is called for each image
        # y_true.shape = 1,1,H,W(,D)
        # Its channels is 1 because it's not one-hot encoded because sometimes
        # I want to know where is certain class to ignore it.

        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        dice_foreground = []
        for b in range(y_pred.shape[0]):
            for c in range(1, y_pred.shape[1]):
                dice_foreground.append( metric.dc(y_pred[b, c], y_true[b,0]==c) )
        return torch.tensor([dice_foreground])

    def aggregate(self, reduction: MetricReduction | str | None = None
                       ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        #from IPython import embed; embed(); asd
        # This will be problematic with multiple classes, but it's easy to fix
        return torch.cat(self._buffers[0], dim=0)
        #return self.get_buffer()

class clDiceMetric(CumulativeIterationMetric):
    # Supposedly works with multiple batch sizes and classes

    def __init__(self) -> None:
        super().__init__()
        self._buffers = None

    def _cl_score(self, segmentation, skeleton):
        """Computes the skeleton volume overlap.
        """
        if isinstance(segmentation, torch.Tensor):
            segmentation = segmentation.cpu().detach().numpy()
        return np.sum(segmentation*skeleton)/np.sum(skeleton)

    def _compute_tensor(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_pred.shape = 1,C,H,W(,D); y_pred \in {0,1}
        # It's 1 instead of B because this function is called for each image
        # y_true.shape = 1,1,H,W(,D)
        # Its channels is 1 because it's not one-hot encoded because sometimes
        # I want to know where is certain class to ignore it.

        #from IPython import embed; embed(); asd

        clDice_foreground = []
        for b in range(y_pred.shape[0]): # For each image
            for c in range(1, y_pred.shape[1]): # For each class

                #tprec and tsens are the opposite, according to the paper
                tprec = self._cl_score(y_pred[b,c],
                                       skeletonize(y_true[b,0]==c))
                tsens = self._cl_score(y_true[b,0]==c,
                                       skeletonize(y_pred[b,c]))

                clDice_foreground.append( 2*tprec*tsens/(tprec+tsens) )

        return torch.tensor([clDice_foreground])

    def aggregate(self, reduction: MetricReduction | str | None = None
                       ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # This will be problematic with multiple classes, but it's easy to fix
        return torch.cat(self._buffers[0], dim=0)

class BettiErrorMetric(CumulativeIterationMetric):

    def __init__(self) -> None:
        super().__init__()
        self._buffers = None

    def _compute_tensor(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()

        betti_err = []
        for b in range(y_pred.shape[0]):
            for c in range(1, y_pred.shape[1]):
                betti_true = self._compute_betti(y_true[b, 0]==c)
                betti_pred = self._compute_betti(y_pred[b, c])
                diff_tmp = np.abs(np.array(betti_true[:-1]) - np.array(betti_pred[:-1]))
                betti_err.append(diff_tmp)
        # B,N,3 (batch, num_patches, betti numbers)
        return torch.tensor([betti_err])

    def _compute_betti(self, patch: np.array) -> List[int]:
        # Compute betti numbers
        cc = gd.CubicalComplex(top_dimensional_cells=1-patch)
        cc.compute_persistence()
        bnum = cc.persistent_betti_numbers(np.inf, -np.inf)
        return bnum

    def aggregate(self, reduction: MetricReduction | str | None = None
                       ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # This will be problematic with multiple classes, but it's easy to fix
        return torch.cat(self._buffers[0], dim=0)

class BettiErrorLocalMetric(CumulativeIterationMetric):

    def __init__(self) -> None:
        super().__init__()
        self._buffers = None
        self.window_size = 128

    def _compute_tensor(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        w = self.window_size

        betti_err = []
        for b in range(y_pred.shape[0]):
            for c in range(1, y_pred.shape[1]):
                y_t = y_true[b, 0]==c
                y_p = y_pred[b, c]
                comb = itertools.product(*[range(0, s, w) for s in y_t.shape])
                tmp_list = []
                for y, x in comb:
                    betti_true = self._compute_betti(y_t[y:y+w, x:x+w])
                    betti_pred = self._compute_betti(y_p[y:y+w, x:x+w])
                    diff_tmp = np.abs(np.array(betti_true[:-1]) - np.array(betti_pred[:-1]))
                    tmp_list.append(diff_tmp)
                betti_err.append( np.mean(tmp_list, axis=0) )
                #from IPython import embed; embed(); asd
        # B,N,3 (batch, num_patches, betti numbers)
        return torch.tensor([betti_err])

    def _compute_betti(self, patch: np.array) -> List[int]:
        # Compute betti numbers
        cc = gd.CubicalComplex(top_dimensional_cells=1-patch)
        cc.compute_persistence()
        bnum = cc.persistent_betti_numbers(np.inf, -np.inf)
        return bnum

    def aggregate(self, reduction: MetricReduction | str | None = None
                       ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # This will be problematic with multiple classes, but it's easy to fix
        return torch.cat(self._buffers[0], dim=0)

