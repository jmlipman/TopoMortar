from typing import List, Tuple, Union, Dict, Callable
from pydoc import locate
from datetime import datetime
from yaml.loader import SafeLoader
import argparse, os, yaml, re, inspect, monai, shutil, itertools
from collections import abc
from monai.transforms import Compose, Randomizable
from torch.optim.lr_scheduler import _LRScheduler
from pathlib import Path
from types import ModuleType
import numpy as np
import pandas as pd
from monai.data.meta_tensor import MetaTensor
import torch
from PIL import Image
import nibabel as nib
import importlib.machinery
import inspect, json, ast

def parseConfig() -> Dict:
    """
    Reads the configuration files. First the paths, then the general
    configuration file (if it exists), and then the specific configuration
    file (if it exists).

    Returns:
      Configuration.
    """
    def parseArguments() -> Dict:
        """Parses and sanitizes the user's given input.
        """
        allowed_datasets = [x[:-3] for x in os.listdir("lib/data") if x.endswith(".py")]

        module = importlib.machinery.SourceFileLoader("losses", "lib/losses.py").load_module()
        class_members = inspect.getmembers(module, inspect.isclass)
        allowed_losses = [m[0] for m in class_members if not "torch" in str(m[1]) and not "monai" in str(m[1])]

        parser = argparse.ArgumentParser(description="TopoMortar Framework")

        parser.add_argument("--config",
                            help="Location of the exp. conf. file",
                            required=False)

        parser.add_argument("--model_state",
                            help="Location of the saved model",
                            required=False)
        parser.add_argument("--output",
                            help="Folder where the output will be saved",
                            required=False)
        parser.add_argument("--run_measure",
                            help="Submit a separate job for computing metrics",
                            required=False,
                            choices=["0", "1"])
        parser.add_argument("--training_set_size",
                            required=False,
                            choices=["small", "large"])
        parser.add_argument("--labels",
                            required=False,
                            choices=["pseudo", "accurate", "noisy"])
        parser.add_argument("--framework",
                            required=False,
                            choices=["supervised", "selfdistillation", "randhue"])
        parser.add_argument("--loss",
                            required=False,
                            choices=allowed_losses) # Get this automatically
        parser.add_argument("--loss_params",
                            required=False,
                            type=json.loads)
        parser.add_argument("--random_seed",
                            required=False,
                            default=42)

        args = parser.parse_args()

        if args.model_state and not Path(args.model_state).is_file():
            msg = f("The specified --model_state `{args.model_state}` "
                    "does not exist.")
            raise ValueError(msg)
        return args

    def update_or_add_config(cfg, specific_conf):

        append_config = ["transform_train", "transform_val",
                "transform_test", "transform_post", "transform_measure",
                "transform_intensity"]

        do_not_update = ['path_base', 'path_general_conf',
                'path_split', 'path_specific_conf', 'path_exp',
                'computer_name']

        if specific_conf.get('full_conf', False):
            # Delete certain things that will be regenerated
            for c in do_not_update:
                if c in specific_conf:
                    del specific_conf[c]
        else:
            for c in append_config:
                if c in cfg and c in specific_conf:
                    cfg[c].extend(specific_conf[c])
                    del specific_conf[c]
        cfg.update(specific_conf)

        return cfg

    params = parseArguments()

    # 1) If the output folder is specified, save results there.
    # Otherwise, use default configuration
    cfg = {'computer_name': getComputerName()}
    if params.output:
        cfg['path_base'] = params.output

    # 2) Read configuration file
    if params.config and Path(params.config).is_file():
        # Load specified yaml file
        config_file = params.config
    else:
        config_file = "config/template_topomortar.yaml"

    # Loading config file
    with open(config_file, 'r') as f:
        cfg_tmp = yaml.load(f, Loader=SafeLoader)
    cfg['path_specific_conf'] = config_file
    cfg = update_or_add_config(cfg, cfg_tmp)

    # 3) Read configuration from the specified parameters
    if params.loss:
        # Parse parameters of the loss function
        if params.loss_params:
            for k in params.loss_params:
                try:
                    params.loss_params[k] = ast.literal_eval(params.loss_params[k])
                except:
                    params.loss_params[k] = params.loss_params[k]
        else:
            params.loss_params = {} # Use default hyperparameters
        cfg['loss'] = {'name': f'lib.losses.{params.loss}',
                        'params': params.loss_params }
    if params.framework:
        cfg['framework'] = params.framework
    if params.random_seed:
        cfg['random_seed'] = int(params.random_seed)

    if not 'dataset' in cfg:
        cfg['dataset'] = {'name': f'lib.data.TopoMortar',
                          'params': {'labels': params.labels,
                                     'training_set_size': params.training_set_size}}

    if "MSETopoWinLoss" in cfg['loss']['name']:
        # This loss function expects a single out channel, the distances
        cfg['model']['params']['out_channels'] = 1

        cfg['transform_post_val'] = []

        cfg['transform_post_val'].append(
                {'name': 'monai.transforms.AsDiscrete', 'params': {'threshold': 1e-5} } )
        cfg['transform_post_val'].append(
                {'name': 'monai.transforms.AsDiscrete', 'params': {'to_onehot': 2} } )

    if "path_base" in cfg: # Training
        cfg['path_exp'] = Path(cfg['path_base']) / cfg['framework'] / cfg['dataset']['params']['labels'] / cfg['dataset']['params']['training_set_size'] / cfg['loss']['name'].split(".")[-1]
    else: # Inference, where the path is already in the --config file
        cfg['path_exp'] = Path(cfg['path_specific_conf']).parents[1]

    if cfg["dataset"]["params"]["training_set_size"] == "large":
        cfg['path_split'] = "dataset/splits.yaml"
    else:
        cfg['path_split'] = "dataset/splits_small.yaml"

    # 4) Sanitize, verification
    if params.model_state:
        cfg['model_state'] = params.model_state
    elif 'model_state' in cfg and not Path(cfg['model_state']).is_file():
        msg = (f"The model_state `{cfg['model_state']}` specified "
                "in the configuration files does not exist.")
        raise ValueError(msg)

    if not 'params' in cfg['loss'] or ('params' in cfg['loss'] and cfg['loss']['params'] is None):
        cfg['loss']['params'] = {}

    # Check that the following configuration is in 'cfg' dictionary
    info = ['path_split', 'path_exp', 'framework',
            'iteration_start', 'batch_size_val',
            'transform_train', 'transform_val', 'transform_test',
            'transform_post_val', 'transform_post_pred',
            'optimizer', 'random_seed', 'callbacks', 'iterations',
            'val_interval', 'metrics_val', 'metrics_pred',
            'computer_name', 'fold', 'move_config_file']
    required_not_given_config = set(info).difference(set(cfg.keys()))
    if len(required_not_given_config):
        msg = ("The following information is required and was not given to "
                f"the cfg dictionary: {required_not_given_config}")
        raise ValueError(msg)

    return cfg


def evaluateTransforms(cfg: dict, splits: List[str]):
    monai.utils.misc.set_determinism(seed=cfg['random_seed'])
    for data_split in splits:
        l = []
        for trans in cfg[f'transform_{data_split}']:
            t = locate(trans['name'])(**trans['params'])
            if isinstance(t, Randomizable):
                t.set_random_state(cfg['random_seed'])
            l.append(t)
        cfg[f'transform_{data_split}'] = Compose(l)
    return cfg

def evaluateConfig(cfg):
    """Convert the str corresponding to functions to actual functions.
    """
    def _evaluateLambdas(cfg: dict) -> dict:
        for k, v in cfg.copy().items():
            if isinstance(v, str) and v.startswith('lambda '):
                v = eval(v)
            cfg.pop(k)
            cfg[k] = v
            if isinstance(v, dict):
                _evaluateLambdas(v)
        return cfg

    def _evaluateMetrics(cfg: dict):
        for mm in ['metrics_val', 'metrics_pred']:
            metrics = []
            for m in cfg[mm]:
                if 'params' in m:
                    metrics.append( locate(m['name'])(**m['params']) )
                else:
                    metrics.append( locate(m['name'])() )
            cfg[mm] = metrics
        return cfg

    def _evaluatePaths(cfg: dict):
        for c in cfg.keys():
            if c.startswith('path') and not cfg[c] is None:
                cfg[c] = Path(cfg[c])
        #cfg['path_datalib'] = Path('lib') / 'data' / f"{cfg['dataset']}.py"
        cfg['path_datalib'] = Path(f"{cfg['dataset']['name'].replace('.', '/')}.py")
        if 'model_state' in cfg:
            cfg['model_state'] = Path(cfg['model_state'])
        return cfg

    cfg = _evaluateLambdas(cfg)
    splits = ['train', 'val', 'test', 'post_val', 'post_pred',
              'measure']
    if cfg['framework'] == "meanteacher":
        splits.append('intensity')
    cfg = evaluateTransforms(cfg, splits)
    cfg = _evaluateMetrics(cfg)
    cfg = _evaluatePaths(cfg)
    if 'params' in cfg['loss'] and 'weight' in cfg['loss']['params']:
        cfg['loss']['params']['weight'] = torch.tensor(cfg['loss']['params']['weight'])
    return cfg

def log(text: str, path: str=None) -> None:

    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    text = date + ': ' + str(text)
    print(text)
    if path:
        with open(Path(path) / 'log.txt', 'a') as f:
            f.write(f'{text}\n')



def callCallbacks(callbacks: List[Callable], prefix: str,
        allvars: dict) -> None:
    """
    Call all callback functions starting with a given prefix.
    Check which inputs the callbacks need, and, from `allvars` (that contains
    locals()) pass those inputs.
    Read more about callback functions in lib.callback

    Args:
      `callbacks`: List of callback functions.
      `prefix`: Prefix of the functions to be called.
      `allvars`: locals() containing all variables in memory.
    """
    for c in callbacks:
        lib, cname = c.rsplit(".", 1)
        c = getattr(__import__(lib, fromlist=[cname]), cname)
        if cname.startswith(prefix):
            input_params = inspect.getfullargspec(c).args
            required_params = {k: allvars[k] for k in input_params}
            c(**required_params)

def safeCopy(source: Path, dest: Path) -> Path:
    """
    Copy the file `source` to `dest`. If the file already exists, rename it.
    """

    destFolder = dest.parents[0]
    if dest.exists():
        r = re.compile(dest.stem + "(_[0-9]+)")
        fi = list(filter(r.match, [x.name for x in destFolder.glob('*')]))
        c = len(fi)+1
        #from IPython import embed; embed(); asd
        destFile = destFolder / f'{dest.stem}_{c}{dest.suffix}'
    else:
        destFile = dest

    shutil.copy(source, destFile)
    return destFile

def saveMetrics(metrics: List[monai.metrics.CumulativeIterationMetric],
                subjects: List[str],
                dataobj: object,
                path_output: Path) -> str:
    # NOTE: This is prepared for one-class problems. For two-class problems
    # we probably need to edit the "aggregate" function in each metric class
    # in lib/metric.py
    all_metrics, metric_names = [], []
    val_str = ""
    for m in metrics:
        tmp_metric = m.aggregate().cpu().detach().numpy()
        if len(tmp_metric.shape) == 1:
            tmp_metric = np.expand_dims(tmp_metric, 1)
        all_metrics.append(tmp_metric)
        metric_names.append( m.__class__.__name__.replace("Metric", "") )
        if metric_names[-1] == "BettiError":
            metric_names[-1] = "Betti0Error"
            metric_names.append("Betti1Error")
            if dataobj.dim == 3:
                metric_names.append("Betti2Error")
        if metric_names[-1] == "BettiErrorLocal":
            metric_names[-1] = "Betti0ErrorLocal"
            metric_names.append("Betti1ErrorLocal")
            if dataobj.dim == 3:
                metric_names.append("Betti2ErrorLocal")
        m.reset()
        val_str += f"Val {metric_names[-1]}: {all_metrics[-1].mean(axis=0)}. "

    #from IPython import embed; embed(); asd
    subjects = np.array([subjects]).T
    #print(subjects)
    #print(all_metrics)
    res = np.concatenate([subjects] + all_metrics, axis=1)
    combination = list(itertools.product(metric_names, dataobj.classes))
    cols = ["ID"] + ["_".join(t) for t in combination]
    df = pd.DataFrame(res, columns=cols)
    # Sort columns
    sorted_cols_names = sorted(cols)
    sorted_cols_names.remove("ID")
    sorted_cols_names = ["ID"] + sorted_cols_names
    df = df[sorted_cols_names]
    df.to_csv(path_output, index=False)
    print(f"Results saved in: {path_output}")
    return val_str

class DataLoaderWrapper:
    """
    For some reason, MONAI doesn't enforce the batch size when using
    num_samples. Read more here:
    https://github.com/Project-MONAI/tutorials/discussions/1244
    So, I created this wrapper to ensure that the batch size is correct.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.iter = self.dataset.__iter__()
        self.batch = None
        self.c = 0 # cursor
        self.bs = self.dataset.batch_size

    def __iter__(self):
        if not self.batch:
            try:
                self.batch = next(self.iter)
            except:
                self.iter = self.dataset.__iter__()
                self.batch = next(self.iter)
        d = {}
        for k in self.batch:
            if isinstance(self.batch[k], dict):
                d[k] = {}
                for t in self.batch[k].keys():
                    d[k][t] = self.batch[k][t][self.c:self.c+self.bs]
            elif isinstance(self.batch[k], MetaTensor): # Tensor
                d[k] = self.batch[k][self.c:self.c+self.bs]
                self.max = self.batch[k].shape[0]
            elif isinstance(self.batch[k], list): # List (of, e.g., str)
                d[k] = self.batch[k][self.c:self.c+self.bs]
                self.max = len(self.batch[k])
            else:
                raise Exception(f"Unknown value `{type(self.batch[k])}`")

        self.c += self.bs
        if self.c >= self.max:
            self.c = 0
            self.batch = None
        yield d

def getComputerName() -> str:
    """Returns the computer's name.
    """
    pc_name = os.uname()[1]
    if pc_name.startswith("n-"): # Node
        pc_name = "DTUcluster"
    elif pc_name.startswith("comp-gpu"): # Node
        pc_name = "titans"
    return pc_name

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def IoU_metric(pred, target):
    inter = np.sum((pred==1)*(target==1), axis=tuple(range(1, target.ndim))) # (B,)
    union = np.sum((pred==1)+(target==1), axis=tuple(range(1, target.ndim))) # (B,)
    iou = np.mean(inter/union)
    return iou


#### Functions to save the images during the evaluation
def save_predictions_PNG_binary(pred: np.ndarray, outputPath: Path,
        subjects_id: List[str]) -> None:

    for i in range(len(subjects_id)):
        outputFilepath = outputPath / f'{subjects_id[i]}.png'
        if pred[i].shape[0] == 2:
            soft_pred = (softmax(pred[i])[1] > 0.5)
        else:
            soft_pred = (pred[i][0] > 0)

        pred_tmp = np.flipud(np.rot90(soft_pred))
        Image.fromarray(np.uint8(pred_tmp[:, :]*255), 'L').save(outputFilepath)

def save_predictions_npy(pred: np.ndarray, outputPath: Path,
        subjects_id: List[str]) -> None:

    for i in range(len(subjects_id)):
        subjects_id[i] = subjects_id[i].replace(".nii", "")
        outputFilepath = outputPath / f'{subjects_id[i]}.npy'
        if pred[i].shape[0] == 2:
            np.save(outputFilepath, softmax(pred[i]))
        else: # MSE, so, shape is 1
            np.save(outputFilepath, pred[i]>0 )

def save_predictions_npy_hardmaxpreserve(pred: np.ndarray, outputPath: Path,
        subjects_id: List[str]) -> None:
    # pred.shape = B,C,H,W(,D)

    for i in range(len(subjects_id)):
        outputFilepath = outputPath / f'{subjects_id[i]}.npy'
        C = pred[i].shape[0]
        argmaxed = np.argmax(pred[i], axis=0)
        tmp_res = np.stack([1.0*(argmaxed==j) for j in range(C)], axis=0)
        np.save(outputFilepath, tmp_res)

