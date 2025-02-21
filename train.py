import lib.utils as utils
from pydoc import locate
from monai.data import DataLoader, CacheDataset, Dataset
from datetime import datetime
import yaml, torch, importlib, time, monai
from lib.core import (
        evaluation,
        trainer_supervised,
        trainer_selfdistillation,
        )

import numpy as np

import pathlib, os
from pathlib import Path
from lib.transforms import RandHued

if __name__ == '__main__':
    t0 = time.time()

    # Gather all configuration-related information
    cfg = utils.parseConfig()

    # Create output folder if it doesn't exist, and find 'run ID'
    run_folders = []
    for x in Path(cfg['path_exp']).glob('*'):
        if x.name.isdigit():
            run_folders.append(int(x.name))
    run_id = max(run_folders)+1 if len(run_folders) > 0 else 1
    cfg['exp_run'] = run_id
    cfg['path_exp'] = str(Path(cfg['path_exp']) / str(run_id))
    path_exp = Path(cfg['path_exp'])
    path_exp.mkdir(parents=True, exist_ok=True)

    # Save the configuration and create the appropriate files
    (path_exp / 'val_scores').mkdir()
    (path_exp / 'models').mkdir()
    #cfg['path_datalib'] = Path('lib') / 'data' / f"{cfg['dataset']}.py"
    with open((path_exp / 'config.yaml'), 'w') as f:
        f.write(yaml.dump(cfg))

    # Evaluate lambda and transform functions (i.e., from str to fun)
    # This is done outside parseConfig() because it's easier to save the
    # functions as str.
    cfg = utils.evaluateConfig(cfg)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    monai.utils.misc.set_determinism(seed=cfg['random_seed'])

    utils.log(f"Experiment: {cfg['path_exp']}", cfg['path_exp'])

    # Preparing the data
    utils.log("Loading data...", cfg['path_exp'])
    tmp_path = str(cfg['path_datalib'].parents[0]).replace(pathlib.os.sep, '.')
    tmp_path += '.' + cfg['path_datalib'].stem
    dataclass = getattr(importlib.import_module(tmp_path), cfg['path_datalib'].stem)
    dataobj = dataclass(**cfg['dataset']['params'])
    trainFiles, valFiles, _ = dataobj.split(cfg['path_split'], fold=cfg['fold'])
    tr_data = Dataset(data=trainFiles, transform=cfg['transform_train'])
    tr_loader = DataLoader(tr_data, batch_size=cfg['batch_size'],
            shuffle=True, num_workers=4)
    tr_loader = utils.DataLoaderWrapper(tr_loader)
    if len(valFiles) > 0:
        val_data = Dataset(data=valFiles, transform=cfg['transform_val'])
        val_loader = DataLoader(val_data, batch_size=cfg['batch_size_val'],
                shuffle=False, num_workers=4)
    else:
        val_loader = []
    utils.log(f"Training images: {len(trainFiles)}", cfg['path_exp'])
    utils.log(f"Validation images: {len(valFiles)}", cfg['path_exp'])

    # Network, optimizer, scheduler, loss
    net = locate(cfg['model']['name'])(**cfg['model']['params']).to(cfg['device'])
    if 'model_state' in cfg:
        utils.log(f"Loading model...", cfg['path_exp'])
        params = torch.load(cfg['model_state'], map_location=cfg['device'])
        net.load_state_dict(params)
    param_n = str(sum(p.numel() for p in net.parameters() if p.requires_grad))
    utils.log(f"Number of trainable parameters: {param_n}", cfg['path_exp'])

    opt = locate(cfg['optimizer']['name'])(net.parameters(),
                 **cfg['optimizer']['params'])
    if 'scheduler' in cfg and not cfg['scheduler'] is None:
        scheduler = locate(cfg['scheduler']['name'])(opt, **cfg['scheduler']['params'])
    else:
        scheduler = None

    loss = locate(cfg['loss']['name'])(**cfg['loss']['params'])
    if 'params' in cfg['val_inferer']:
        val_inferer = locate(cfg['val_inferer']['name'])(**cfg['val_inferer']['params'])
    else:
        val_inferer = locate(cfg['val_inferer']['name'])()

    # Which scenario?
    if cfg['framework'] == "selfdistillation":
        trainer = trainer_selfdistillation
        kwargs = {'selfdistillation_iterations': 3}
        dataset_name = cfg['dataset']['name'].split(".")[-1]

        cfg['iterations'] = 4000 # 4000x3 = 12000
        cfg['val_interval'] = 1000
        cfg['scheduler']['params']['lr_lambda'] = lambda ep: (1 - ep/4000)**0.9

    elif cfg['framework'] == "randhue":
        trainer = trainer_supervised
        huetrans = RandHued(keys=["image"], prob=0.5)
        tr_loader.dataset.dataset.transform.transforms = tuple(list(tr_loader.dataset.dataset.transform.transforms) + [huetrans])
        trainer = trainer_supervised
        kwargs = {}

    else: # Standard supervised learning

        trainer = trainer_supervised
        kwargs = {}

    t1 = time.time()
    net = torch.compile(net)
    # Training loop
    trainer(model=net,
            tr_loader=tr_loader,
            loss=loss,
            opt=opt,
            scheduler=scheduler,
            iteration_start=cfg['iteration_start'],
            iterations=cfg['iterations'],
            val_loader=val_loader,
            val_interval=cfg['val_interval'],
            val_inferer=val_inferer,
            metrics=cfg['metrics_val'],
            dataobj=dataobj,
            postprocessing=cfg['transform_post_val'],
            path_exp=cfg['path_exp'],
            device=cfg['device'],
            callbacks=cfg['callbacks'],
            **kwargs)

    msg = f"Total training time - {np.round((time.time()-t1)/3600, 3)} hours"
    utils.log(msg, cfg['path_exp'])

    # Move file from exp_started2 to exp_pred3 folder
    msg = f"Total running time - {np.round((time.time()-t0)/3600, 3)} hours"
    utils.log(msg, cfg['path_exp'])
    utils.log("End", cfg['path_exp'])
