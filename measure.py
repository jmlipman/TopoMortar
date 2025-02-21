from pydoc import locate
import lib.utils as utils
import time, importlib, torch, pathlib, yaml, monai
from monai.data import DataLoader, Dataset
from lib.core import evaluation
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    time.sleep(np.random.random()*10)
    t0 = time.time()

    cfg = utils.parseConfig()
    path_specific_conf = Path(cfg['path_specific_conf'])

    if Path(cfg['path_specific_conf']).name != "config.yaml":
        #print("REMEMBER TO DO THIS")
        path_specific_conf.unlink()
    else:
        cfg['metrics_pred'].append({'name': 'lib.metrics.BettiErrorLocalMetric'})
        cfg['metrics_pred'].append({'name': 'lib.metrics.clDiceMetric'})

    cfg = utils.evaluateConfig(cfg)
    cfg['path_exp'] = cfg['path_exp'] / str(cfg['exp_run'])
    path_output = cfg['path_exp'] / 'results.csv'
    if path_output.is_file():
        msg = f"Output file `{path_output}` already exists."
        raise ValueError(msg)

    utils.log(f"Measure metrics - {cfg['path_exp']}", cfg['path_exp'])

    # Loading the data
    tmp_path = str(cfg['path_datalib'].parents[0]).replace(pathlib.os.sep, '.')
    tmp_path += '.' + cfg['path_datalib'].stem
    #datalib = importlib.import_module(tmp_path)
    dataclass = getattr(importlib.import_module(tmp_path), cfg['path_datalib'].stem)
    dataobj = dataclass(**cfg['dataset']['params'])
    _, _, testFiles = dataobj.split(cfg['path_split'], fold=cfg['fold'])

    for i in range(len(testFiles)):
        #from IPython import embed; embed(); asd
        #filename = testFiles[i]['id'] + '.' + dataobj.ext
        #testFiles[i]['prediction'] = str(cfg['path_exp'] / 'preds' / filename)
        testFiles[i]['prediction'] = [t for t in (cfg['path_exp'] / 'preds').glob("*") if t.is_file() and t.stem == testFiles[i]["id"]][0]
        del testFiles[i]['image']

    test_data = Dataset(data=testFiles, transform=cfg['transform_measure'])
    test_loader = DataLoader(test_data, batch_size=cfg['batch_size_val'],
            shuffle=False, num_workers=1)
    utils.log(f"Test images: {len(testFiles)}", cfg['path_exp'])
    if len(testFiles) == 0:
        exit()


    metrics = cfg['metrics_pred']
    all_subjects_id = []
    with torch.no_grad():
        for eval_i, eval_data in enumerate(test_loader):
            print(f"Measuring image batch {eval_i+1}/{len(test_loader)}")
            #if not "6336" in eval_data["id"]:
            #    continue
            #from IPython import embed; embed(); asd
            Y = eval_data["label"]
            y_pred = eval_data["prediction"]
            subjects_id = eval_data['id']
            all_subjects_id.extend(subjects_id)
            #from IPython import embed; embed(); asd
            for metric in metrics:
                metric(y_pred=y_pred, y=Y)

    utils.saveMetrics(metrics, all_subjects_id, dataobj, path_output)

    msg = f"Total measure time - {np.round((time.time()-t0)/3600, 3)} hours"
    utils.log(msg, cfg['path_exp'])
