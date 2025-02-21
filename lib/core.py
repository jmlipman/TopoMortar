import torch, monai, time
import lib.utils as utils
from monai.inferers import Inferer
from monai.data import decollate_batch
from typing import List
from types import ModuleType
from torch.nn.modules.loss import _Loss as Loss
from torch.cuda import amp
from pathlib import Path
import numpy as np
from datetime import datetime

from typing import List, Callable, Union, Optional
from types import ModuleType
from torch.nn.modules.loss import _Loss as Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
import copy
import torch.nn.functional as F

def trainer_selfdistillation(model: torch.nn.Module,
            tr_loader: monai.data.dataloader.DataLoader,
            loss: Loss,
            opt: Optimizer,
            scheduler: Union[_LRScheduler, None],
            iteration_start: int,
            iterations: int,
            val_loader: monai.data.dataloader.DataLoader,
            val_interval: int,
            val_inferer: Inferer,
            metrics: List[monai.metrics.CumulativeIterationMetric],
            dataobj: object,
            postprocessing: monai.transforms.Compose,
            path_exp: Path,
            device: str,
            callbacks: List[Callable]=[],
            **kwargs) -> None:

    def weight_reset(m: torch.nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    selfdistillation_iterations = kwargs["selfdistillation_iterations"]
    non_aug_tr_dataset = copy.deepcopy(tr_loader.dataset)
    non_aug_trans = copy.deepcopy([t for t in tr_loader.dataset.dataset.transform.transforms if not "Rand" in str(t.__class__)])
    non_aug_tr_dataset.dataset.transform.transforms = tuple(non_aug_trans)

    saving_fun = utils.save_predictions_npy

    for it in range(1, selfdistillation_iterations+1):
        if it == 2:
            # Load a new trainer on the pseudo-labels
            for i in range(len(tr_loader.dataset.dataset.data)):

                #from IPython import embed; embed(); asd
                tmp_newname = tr_loader.dataset.dataset.data[i]["label"].name.replace(".png", ".npy")
                tmp_newpath = path_exp / 'newlabels' / 'preds' / str(it-1) / tmp_newname
                tr_loader.dataset.dataset.data[i]["label"] = tmp_newpath

            tr_loader.dataset.dataset.transform.transforms[1].keys = ['image']
            loss.to_onehot = False

            if "RegionWise" in str(loss):
                loss.compute_rwmap = loss._compute_rrwmap_selfdistillation

        # Train a model
        trainer_supervised(model=model,
                tr_loader=tr_loader,
                loss=loss,
                opt=opt,
                scheduler=scheduler,
                iteration_start=iteration_start,
                iterations=iterations,
                val_loader=val_loader,
                val_interval=val_interval,
                val_inferer=val_inferer,
                metrics=metrics,
                dataobj=dataobj,
                postprocessing=postprocessing,
                path_exp=path_exp,
                device=device,
                callbacks=callbacks,
                kwargs={})

        model.eval()

        # Saving the predictions as "newlabels"
        val_str = evaluation(model=model,
                             data_loader=non_aug_tr_dataset,
                             iteration=it,
                             batch_size=tr_loader.bs,
                             inferer=val_inferer,
                             metrics=[],
                             dataobj=dataobj,
                             path_exp=path_exp / 'newlabels',
                             device=device,
                             postprocessing=postprocessing,
                             loss=None,
                             callbacks=callbacks,
                             save_preds=saving_fun)
        model.train()
        opt = torch.optim.SGD(model.parameters(), **opt.defaults)
        model.apply(fn=weight_reset)


def trainer_supervised(model: torch.nn.Module,
            tr_loader: monai.data.dataloader.DataLoader,
            loss: Loss,
            opt: Optimizer,
            scheduler: Union[_LRScheduler, None], # CONFIRM THIS
            iteration_start: int,
            iterations: int,
            val_loader: monai.data.dataloader.DataLoader,
            val_interval: int,
            val_inferer: Inferer,
            metrics: List[monai.metrics.CumulativeIterationMetric],
            dataobj: object,
            postprocessing: monai.transforms.Compose,
            path_exp: Path,
            device: str,
            callbacks: List[Callable]=[],
            **kwargs) -> None:

    utils.log("Start training", path_exp)
    t0 = time.time()
    it = iteration_start
    scaler = amp.GradScaler()

    # As some servers only allow you to run jobs for max. 3 days, it
    # can be important to resume the training and re-execute certain
    # procedures, like scheduler.step()
    utils.callCallbacks(callbacks, '_start_training', locals())

    tr_loss = 0 # exponential moving average (alpha=0.99)
    model.train()
    while it <= iterations:

        utils.callCallbacks(callbacks, '_start_iteration', locals())
        for tr_i, batch_data in enumerate(tr_loader):
            X, Y = (
                    batch_data['image'].to(device),
                    batch_data['label'].to(device),
            )
            #print(X.shape, Y.shape)
            utils.callCallbacks(callbacks, '_start_train_iteration', locals())

            with amp.autocast():
                y_pred = model(X)

                if (hasattr(loss, "activate") and
                        it > iterations*loss.activate_when):
                    loss.activate = True

                # Assumption: DynUNet and Deep supervision.
                # So, output = torch.Size([B, DS, C, H, W, D])
                tr_loss_tmp = loss(y_pred[:, 0], Y)

                if hasattr(loss, "activate"):
                    loss.activate = False
                weights = np.array([1/(2**i) for i in range(7)])
                weights = weights / np.sum(weights)
                for i in range(1, y_pred.shape[1]):
                    tr_loss_tmp += loss(y_pred[:, i], Y)*weights[i]
                tr_loss_tmp /= y_pred.shape[1]

            tr_loss = 0.99*tr_loss + 0.01*tr_loss_tmp.cpu().detach().numpy()

            # Optimization
            opt.zero_grad()
            scaler.scale(tr_loss_tmp).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            utils.callCallbacks(callbacks, '_after_compute_grads', locals())
            scaler.step(opt)
            scaler.update()

            utils.callCallbacks(callbacks, '_end_train_iteration', locals())

            if it % val_interval == 0:
                val_str = ""
                if len(val_loader) > 0:
                    utils.log("Validation", path_exp)
                    model.eval()

                    val_str = evaluation(model=model,
                                         data_loader=val_loader,
                                         iteration=it,
                                         batch_size=tr_loader.bs,
                                         inferer=val_inferer,
                                         metrics=metrics,
                                         dataobj=dataobj,
                                         path_exp=path_exp,
                                         device=device,
                                         postprocessing=postprocessing,
                                         loss=None,
                                         callbacks=callbacks,
                                         save_preds=dataobj.save_predictions_fn)
                    model.train()

                eta = time.time() + (iterations-it)*(time.time()-t0)/it
                eta = datetime.fromtimestamp(eta).strftime('%Y-%m-%d %H:%M:%S')
                msg = f"Iteration: {it}. Loss: {tr_loss}. {val_str} ETA: {eta}"
                utils.log(msg, path_exp)

            if scheduler:
                scheduler.step()
            it += 1

            if it > iterations:
                break


    utils.callCallbacks(callbacks, "_end_training", locals())

def evaluation(model: torch.nn.Module,
               data_loader: monai.data.dataloader.DataLoader,
               iteration: int,
               batch_size: int,
               inferer: Inferer,
               metrics: List[monai.metrics.CumulativeIterationMetric],
               dataobj: ModuleType,
               path_exp: Path, # Folder
               device: str,
               postprocessing: monai.transforms.compose.Compose,
               loss: Loss=None,
               callbacks: List[str]=[],
               save_preds: Union[Callable,None]=None) -> str:
    """
    Performs the prediction, computes the metrics, optionally saves the preds.
    """

    val_loss, all_subjects_id = 0, []
    with torch.no_grad():
        for val_i, val_data in enumerate(data_loader):

            utils.callCallbacks(callbacks, "_start_val_subject", locals())
            print(f"Predicting image {val_i+1}/{len(data_loader)}")

            #print(val_data["id"])
            X = val_data["image"].to(device) # cuda
            if loss or len(metrics) > 0:
                Y = val_data["label"] # cpu

            subjects_id = [val_data["id"][i] for i in range(len(val_data['id']))]
            all_subjects_id.extend(subjects_id)

            y_pred = inferer(inputs=X, network=model).to("cpu")

            # Note: I moved this up here because the "postprocessing"
            # done below will inevitably argmax my preds, which is necessary
            # for the metrics.
            if save_preds: # Only works if it's not None

                path_preds = path_exp / 'preds' / str(iteration)
                path_preds.mkdir(parents=True, exist_ok=True)
                print("Save prediction in: ", path_preds)
                save_preds(pred=y_pred.cpu().detach().numpy(),
                                       outputPath=path_preds,
                                       subjects_id=subjects_id)


            if loss:
                val_loss += loss(y_pred, Y) / len(data_loader)
            y_pred = [postprocessing(i) for i in decollate_batch(y_pred)]

            for metric in metrics:
                metric(y_pred=y_pred, y=Y) # B,H,W(,D)

            utils.callCallbacks(callbacks, "_end_val_subject", locals())

    val_str = ""
    if loss:
        val_str = f"Val loss: {val_loss}. "

    if len(metrics) == 0:
        return val_str

    # Save metrics
    val_str += utils.saveMetrics(metrics, all_subjects_id, dataobj,
            path_exp / 'val_scores' / f'scores-{iteration}.csv')

    return val_str

