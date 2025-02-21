import numpy as np
from pathlib import Path
import pandas as pd

TYPES_PATH = "../dataset/dataset_types.txt"
NewColsNames = {"Betti0ErrorLocal_mortar":"Betti0Local", "Betti0Error_mortar":"Betti0", "Betti1ErrorLocal_mortar":"Betti1Local", "Betti1Error_mortar":"Betti1", "Dice_mortar":"Dice", "HD95_mortar": "HD95", "clDice_mortar":"clDice"}
Metrics = ["Betti0", "Betti1", "Dice", "HD95"]
NewLossNames = {"CEDiceLoss": "CEDice", "RegionWiseLoss": "RWLoss",
        "TopoLoss": "TopoLoss", "MSETopoWinLoss": "TOPO",
        "clDiceLoss": "clDice", "WarpingLoss": "Warping",
        "CESkeletonRecallLoss": "SkelRecall", "cbDiceLoss": "cbDice"}
PRECISION = 2 # Number of decimals
def r(x):
    return np.round(x, PRECISION)

def isBetter(g1, g2, metric):
    """Is g1 better than g2?
    """
    if metric in ["Betti0", "Betti1", "HD95"]:
        return r(np.mean(g1)) < r(np.mean(g2))
    else:
        return r(np.mean(g1)) > r(np.mean(g2))

def nonparametric_permutationtest(A, B, perms=10000):

    stat = np.abs(np.mean(A) - np.mean(B))
    p = 0
    for _ in range(perms):
        swap = (np.random.random(len(A)) > 0.5).astype(int)
        if np.mean(swap*A+(1-swap)*B) - np.mean((1-swap)*A+swap*B) > stat:
            p += 1
    return p / perms

def gatherData(path):
    configurations = []
    for f_path in Path(path).rglob("*csv"):
        if not str(f_path.parents[0]) in configurations:
            configurations.append(str(f_path.parents[0]))

    # Load all data in a dictionary
    alldata, statistics, significance = {}, {}, {}
    configurations_without_losses = []
    for conf in configurations:
        parts = conf.split("/")
        if not parts[0] in alldata:
            alldata[parts[0]] = {}
            statistics[parts[0]] = {}
            significance[parts[0]] = {}
        if not parts[1] in alldata[parts[0]]:
            alldata[parts[0]][parts[1]] = {}
            statistics[parts[0]][parts[1]] = {}
            significance[parts[0]][parts[1]] = {}
        if not parts[2] in alldata[parts[0]][parts[1]]:
            alldata[parts[0]][parts[1]][parts[2]] = {}
            statistics[parts[0]][parts[1]][parts[2]] = {}
            significance[parts[0]][parts[1]][parts[2]] = {}
        if not parts[3] in significance[parts[0]][parts[1]][parts[2]]:
            significance[parts[0]][parts[1]][parts[2]][parts[3]] = {}

        conf_ = "/".join(parts[:-1])
        if not conf_ in configurations_without_losses:
            configurations_without_losses.append(conf_)

        alldfs_tmp = []
        for i in range(1, 11):
            df = pd.read_csv(TYPES_PATH)
            df_tmp = pd.read_csv(f"{conf}/{i}.csv")
            #df = df.replace(9999999, 722.6631303726516) ### CHECK THIS OUT
            alldfs_tmp.append( df.merge(df_tmp, on="ID").replace(9999999, 722.6631303726516) )
        df = pd.concat(alldfs_tmp)
        df.rename(columns=NewColsNames, inplace=True)
        df.sort_values(by="ID", inplace=True)
        del df["ID"]
        alldata[parts[0]][parts[1]][parts[2]][parts[3]] = df

    # Compute means and stds
    for conf in configurations:
        parts = conf.split("/")
        df = alldata[parts[0]][parts[1]][parts[2]][parts[3]]

        means = df.groupby("group").mean()
        stds = df.groupby("group").std()

        oodmeans = means[means.index.isin(["angles", "colors", "graffiti", "objects", "occlusion", "shadows"])].mean()
        oodstds = stds[means.index.isin(["angles", "colors", "graffiti", "objects", "occlusion", "shadows"])].mean()
        idmeans = means[means.index.isin(["indistr"])]
        idstds = stds[means.index.isin(["indistr"])]

        #allmeans = means.mean()
        #allstds = stds.mean() # Average across groups, std within groups

        #means = pd.concat([means.T, allmeans, oodmeans], axis=1)
        means = pd.concat([idmeans.T, oodmeans], axis=1)
        #stds = pd.concat([stds.T, allstds, oodstds], axis=1)
        stds = pd.concat([idstds.T, oodstds], axis=1)

        means.rename(columns={0: "ood"}, inplace=True)
        stds.rename(columns={0: "ood"}, inplace=True)

        statistics[parts[0]][parts[1]][parts[2]][parts[3]] = {
                "mean": dict(means.round(PRECISION).astype(str)),
                      "std": dict(stds.round(2).astype(str)) }


    # Statistical significance comparing each loss vs. CEDice loss
    #alldata[parts[0]][parts[1]][parts[2]][parts[3]][metric] = 0/1
    for i, conf in enumerate(configurations_without_losses):
        print(f"Computing significance {i+1}/{len(configurations_without_losses)}")
        parts = conf.split("/")
        for loss in NewLossNames.keys():
            if loss == "CEDiceLoss":
                continue
            for metric in Metrics:
                significance[parts[0]][parts[1]][parts[2]]["CEDiceLoss"][metric] = {"indistr": False, "ood": False}

                # In distribution
                idx_id = alldata[parts[0]][parts[1]][parts[2]]["CEDiceLoss"]["group"]=="indistr"
                g1_id = alldata[parts[0]][parts[1]][parts[2]]["CEDiceLoss"][idx_id][metric]
                g2_id = alldata[parts[0]][parts[1]][parts[2]][loss][idx_id][metric]
                pval_id = nonparametric_permutationtest(g1_id, g2_id)

                idx_ood = alldata[parts[0]][parts[1]][parts[2]]["CEDiceLoss"]["group"]!="indistr"
                g1_ood = alldata[parts[0]][parts[1]][parts[2]]["CEDiceLoss"][idx_ood][metric]
                g2_ood = alldata[parts[0]][parts[1]][parts[2]][loss][idx_ood][metric]
                pval_ood = nonparametric_permutationtest(g1_ood, g2_ood)

                significance[parts[0]][parts[1]][parts[2]][loss][metric] = {"indistr": (pval_id < 0.05) & isBetter(g2_id, g1_id, metric), "ood": (pval_ood < 0.05) & isBetter(g2_ood, g1_ood, metric)}

    return statistics, significance


def table1(statistics, significance):
    #from IPython import embed; embed(); asd
    t = "<table>\n"
    t += "<thead>\n"
    t += "<tr>\n"
    t += "\t<th>Loss</th>\n"
    for metric in Metrics:
        t += f"<th>{metric}</th>"
    t += "\n</tr>\n"
    t += "</thead>\n"
    t += "<tbody>\n"

    for loss in NewLossNames:
        t += "<tr>\n"
        t += f"\t<td>{NewLossNames[loss]}</td>"
        for metric in Metrics:
            tmp_mean = statistics["supervised"]["accurate"]["large"][loss]["mean"]["indistr"][metric]
            tmp_std = statistics["supervised"]["accurate"]["large"][loss]["std"]["indistr"][metric]
            if significance["supervised"]["accurate"]["large"][loss][metric]["indistr"]:
                t += f"<td><b>{tmp_mean} ± {tmp_std}</b></td>"
            else:
                t += f"<td>{tmp_mean} ± {tmp_std}</td>"
        t += "\n</tr>\n"
    t += "</tbody>\n"
    t += "</table>\n"

    return t

def table2(statistics, significance):
    t = "<table>\n"
    t += "<thead>\n"
    t += "<tr>\n"
    t += "\t<th>Loss</th>\n"
    for metric in Metrics:
        t += f"<th>{metric}</th>"
    t += "\n</tr>\n"
    t += "</thead>\n"
    t += "<tbody>\n"

    for loss in NewLossNames:
        t += "<tr>\n"
        t += f"\t<td>{NewLossNames[loss]}</td>"
        for metric in Metrics:
            tmp_mean = statistics["supervised"]["accurate"]["large"][loss]["mean"]["ood"][metric]
            tmp_std = statistics["supervised"]["accurate"]["large"][loss]["std"]["ood"][metric]
            if significance["supervised"]["accurate"]["large"][loss][metric]["ood"]:
                t += f"<td><b>{tmp_mean} ± {tmp_std}</b></td>"
            else:
                t += f"<td>{tmp_mean} ± {tmp_std}</td>"
        t += "\n</tr>\n"
    t += "</tbody>\n"
    t += "</table>\n"

    return t

def table3(statistics, significance):
    t = "<table>\n"
    t += "<thead>\n"
    t += "<tr>\n"
    t += "<th></th>\n"
    t += "\t<th>Loss</th>\n"
    for metric in Metrics:
        t += f"<th>{metric}</th>"
    t += "\n</tr>\n"
    t += "</thead>\n"
    t += "<tbody>\n"

    for distr_i, distr_t in zip(["indistr", "ood"], ["ID", "OOD"]):

        for loss_i, loss in enumerate(NewLossNames):
            t += "<tr>\n"
            if loss_i == 0:
                t += f'<td rowspan="{len(NewLossNames)}">{distr_t}</td>'
            t += f"\t<td>{NewLossNames[loss]}</td>"
            for metric in Metrics:
                tmp_mean = statistics["supervised"]["accurate"]["small"][loss]["mean"][distr_i][metric]
                tmp_std = statistics["supervised"]["accurate"]["small"][loss]["std"][distr_i][metric]
                if significance["supervised"]["accurate"]["small"][loss][metric][distr_i]:
                    t += f"<td><b>{tmp_mean} ± {tmp_std}</b></td>"
                else:
                    t += f"<td>{tmp_mean} ± {tmp_std}</td>"
            t += "\n</tr>\n"

    t += "</tbody>\n"
    t += "</table>\n"

    return t

def table4(statistics, significance):
    t = "<table>\n"
    t += "<thead>\n"
    t += "<tr>\n"
    t += "<th></th>\n"
    t += "\t<th>Loss</th>\n"
    for metric in Metrics:
        t += f"<th>{metric}</th>"
    t += "\n</tr>\n"
    t += "</thead>\n"
    t += "<tbody>\n"

    for distr_i, distr_t in zip(["indistr", "ood"], ["ID", "OOD"]):

        for loss_i, loss in enumerate(NewLossNames):
            t += "<tr>\n"
            if loss_i == 0:
                t += f'<td rowspan="{len(NewLossNames)}">{distr_t}</td>'
            t += f"\t<td>{NewLossNames[loss]}</td>"
            for metric in Metrics:
                tmp_mean = statistics["supervised"]["pseudo"]["large"][loss]["mean"][distr_i][metric]
                tmp_std = statistics["supervised"]["pseudo"]["large"][loss]["std"][distr_i][metric]
                if significance["supervised"]["pseudo"]["large"][loss][metric][distr_i]:
                    t += f"<td><b>{tmp_mean} ± {tmp_std}</b></td>"
                else:
                    t += f"<td>{tmp_mean} ± {tmp_std}</td>"
            t += "\n</tr>\n"

    t += "</tbody>\n"
    t += "</table>\n"

    return t

def table5(statistics, significance):
    t = "<table>\n"
    t += "<thead>\n"
    t += "<tr>\n"
    t += "<th></th>\n"
    t += "\t<th>Loss</th>\n"
    for metric in Metrics:
        t += f"<th>{metric}</th>"
    t += "\n</tr>\n"
    t += "</thead>\n"
    t += "<tbody>\n"

    for distr_i, distr_t in zip(["indistr", "ood"], ["ID", "OOD"]):

        for loss_i, loss in enumerate(NewLossNames):
            t += "<tr>\n"
            if loss_i == 0:
                t += f'<td rowspan="{len(NewLossNames)}">{distr_t}</td>'
            t += f"\t<td>{NewLossNames[loss]}</td>"
            for metric in Metrics:
                tmp_mean = statistics["supervised"]["noisy"]["large"][loss]["mean"][distr_i][metric]
                tmp_std = statistics["supervised"]["noisy"]["large"][loss]["std"][distr_i][metric]
                if significance["supervised"]["noisy"]["large"][loss][metric][distr_i]:
                    t += f"<td><b>{tmp_mean} ± {tmp_std}</b></td>"
                else:
                    t += f"<td>{tmp_mean} ± {tmp_std}</td>"
            t += "\n</tr>\n"

    t += "</tbody>\n"
    t += "</table>\n"

    return t

def table6(statistics, significance):
    t = "<table>\n"
    t += "<thead>\n"
    t += "<tr>\n"
    t += "<th colspan=2></th>"
    t += f"<th colspan={len(Metrics)}>Ideal scenario</th>"
    t += "<th>vs.</th>"
    t += f"<th colspan={len(Metrics)}>RandHue</th>"
    t += "</tr>\n"

    t += "<tr>\n"
    t += "<th></th>\n"
    t += "\t<th>Loss</th>\n"
    for metric in Metrics:
        t += f"<th>{metric}</th>"
    t += "<th></th>\n"
    for metric in Metrics:
        t += f"<th>{metric}</th>"
    t += "\n</tr>\n"
    t += "</thead>\n"
    t += "<tbody>\n"

    for distr_i, distr_t in zip(["indistr", "ood"], ["ID", "OOD"]):

        for loss_i, loss in enumerate(NewLossNames):
            t += "<tr>\n"
            if loss_i == 0:
                t += f'<td rowspan="{len(NewLossNames)}">{distr_t}</td>'
            t += f"\t<td>{NewLossNames[loss]}</td>"
            for metric in Metrics:
                tmp_mean = statistics["supervised"]["accurate"]["large"][loss]["mean"][distr_i][metric]
                tmp_std = statistics["supervised"]["accurate"]["large"][loss]["std"][distr_i][metric]
                if significance["supervised"]["accurate"]["large"][loss][metric][distr_i]:
                    t += f"<td><b>{tmp_mean} ± {tmp_std}</b></td>"
                else:
                    t += f"<td>{tmp_mean} ± {tmp_std}</td>"
            t += "<td></td>"
            for metric in Metrics:
                tmp_mean = statistics["randhue"]["accurate"]["large"][loss]["mean"][distr_i][metric]
                tmp_std = statistics["randhue"]["accurate"]["large"][loss]["std"][distr_i][metric]
                if significance["randhue"]["accurate"]["large"][loss][metric][distr_i]:
                    t += f"<td><b>{tmp_mean} ± {tmp_std}</b></td>"
                else:
                    t += f"<td>{tmp_mean} ± {tmp_std}</td>"

            t += "\n</tr>\n"

    t += "</tbody>\n"
    t += "</table>\n"

    return t

def table7(statistics, significance):
    t = "<table>\n"
    t += "<thead>\n"
    t += "<tr>\n"
    t += "<th colspan=2></th>"
    t += f"<th colspan={len(Metrics)}>Small training set</th>"
    t += "<th>vs.</th>"
    t += f"<th colspan={len(Metrics)}>Large training set</th>"
    t += "</tr>\n"

    t += "<tr>\n"
    t += "<th></th>\n"
    t += "\t<th>Loss</th>\n"
    for metric in Metrics:
        t += f"<th>{metric}</th>"
    t += "<th></th>\n"
    for metric in Metrics:
        t += f"<th>{metric}</th>"
    t += "\n</tr>\n"
    t += "</thead>\n"
    t += "<tbody>\n"

    for distr_i, distr_t in zip(["indistr", "ood"], ["ID", "OOD"]):

        for loss_i, loss in enumerate(NewLossNames):
            t += "<tr>\n"
            if loss_i == 0:
                t += f'<td rowspan="{len(NewLossNames)}">{distr_t}</td>'
            t += f"\t<td>{NewLossNames[loss]}</td>"
            for metric in Metrics:
                tmp_mean = statistics["supervised"]["accurate"]["small"][loss]["mean"][distr_i][metric]
                tmp_std = statistics["supervised"]["accurate"]["small"][loss]["std"][distr_i][metric]
                if significance["supervised"]["accurate"]["small"][loss][metric][distr_i]:
                    t += f"<td><b>{tmp_mean} ± {tmp_std}</b></td>"
                else:
                    t += f"<td>{tmp_mean} ± {tmp_std}</td>"
            t += "<td></td>"
            for metric in Metrics:
                tmp_mean = statistics["supervised"]["accurate"]["large"][loss]["mean"][distr_i][metric]
                tmp_std = statistics["supervised"]["accurate"]["large"][loss]["std"][distr_i][metric]
                if significance["supervised"]["accurate"]["large"][loss][metric][distr_i]:
                    t += f"<td><b>{tmp_mean} ± {tmp_std}</b></td>"
                else:
                    t += f"<td>{tmp_mean} ± {tmp_std}</td>"

            t += "\n</tr>\n"

    t += "</tbody>\n"
    t += "</table>\n"

    return t

def table8(statistics, significance):
    t = "<table>\n"
    t += "<thead>\n"
    t += "<tr>\n"
    t += "<th colspan=2></th>"
    t += f"<th colspan={len(Metrics)}>Standard supervised learning + Pseudo-labels</th>"
    t += "<th>vs.</th>"
    t += f"<th colspan={len(Metrics)}>Self-distillation + Pseudo-labels</th>"
    t += "</tr>\n"

    t += "<tr>\n"
    t += "<th></th>\n"
    t += "\t<th>Loss</th>\n"
    for metric in Metrics:
        t += f"<th>{metric}</th>"
    t += "<th></th>\n"
    for metric in Metrics:
        t += f"<th>{metric}</th>"
    t += "\n</tr>\n"
    t += "</thead>\n"
    t += "<tbody>\n"

    for distr_i, distr_t in zip(["indistr", "ood"], ["ID", "OOD"]):

        for loss_i, loss in enumerate(NewLossNames):
            t += "<tr>\n"
            if loss_i == 0:
                t += f'<td rowspan="{len(NewLossNames)}">{distr_t}</td>'
            t += f"\t<td>{NewLossNames[loss]}</td>"
            for metric in Metrics:
                tmp_mean = statistics["supervised"]["pseudo"]["large"][loss]["mean"][distr_i][metric]
                tmp_std = statistics["supervised"]["pseudo"]["large"][loss]["std"][distr_i][metric]
                if significance["supervised"]["pseudo"]["large"][loss][metric][distr_i]:
                    t += f"<td><b>{tmp_mean} ± {tmp_std}</b></td>"
                else:
                    t += f"<td>{tmp_mean} ± {tmp_std}</td>"
            t += "<td></td>"
            for metric in Metrics:
                tmp_mean = statistics["selfdistillation"]["pseudo"]["large"][loss]["mean"][distr_i][metric]
                tmp_std = statistics["selfdistillation"]["pseudo"]["large"][loss]["std"][distr_i][metric]
                if significance["selfdistillation"]["pseudo"]["large"][loss][metric][distr_i]:
                    t += f"<td><b>{tmp_mean} ± {tmp_std}</b></td>"
                else:
                    t += f"<td>{tmp_mean} ± {tmp_std}</td>"

            t += "\n</tr>\n"

    t += "</tbody>\n"
    t += "</table>\n"

    return t

def table9(statistics, significance):
    t = "<table>\n"
    t += "<thead>\n"
    t += "<tr>\n"
    t += "<th colspan=2></th>"
    t += f"<th colspan={len(Metrics)}>Standard supervised learning + Noisy labels</th>"
    t += "<th>vs.</th>"
    t += f"<th colspan={len(Metrics)}>Self-distillation + Noisy labels</th>"
    t += "</tr>\n"

    t += "<tr>\n"
    t += "<th></th>\n"
    t += "\t<th>Loss</th>\n"
    for metric in Metrics:
        t += f"<th>{metric}</th>"
    t += "<th></th>\n"
    for metric in Metrics:
        t += f"<th>{metric}</th>"
    t += "\n</tr>\n"
    t += "</thead>\n"
    t += "<tbody>\n"

    for distr_i, distr_t in zip(["indistr", "ood"], ["ID", "OOD"]):

        for loss_i, loss in enumerate(NewLossNames):
            t += "<tr>\n"
            if loss_i == 0:
                t += f'<td rowspan="{len(NewLossNames)}">{distr_t}</td>'
            t += f"\t<td>{NewLossNames[loss]}</td>"
            for metric in Metrics:
                tmp_mean = statistics["supervised"]["noisy"]["large"][loss]["mean"][distr_i][metric]
                tmp_std = statistics["supervised"]["noisy"]["large"][loss]["std"][distr_i][metric]
                if significance["supervised"]["noisy"]["large"][loss][metric][distr_i]:
                    t += f"<td><b>{tmp_mean} ± {tmp_std}</b></td>"
                else:
                    t += f"<td>{tmp_mean} ± {tmp_std}</td>"
            t += "<td></td>"
            for metric in Metrics:
                tmp_mean = statistics["selfdistillation"]["noisy"]["large"][loss]["mean"][distr_i][metric]
                tmp_std = statistics["selfdistillation"]["noisy"]["large"][loss]["std"][distr_i][metric]
                if significance["selfdistillation"]["noisy"]["large"][loss][metric][distr_i]:
                    t += f"<td><b>{tmp_mean} ± {tmp_std}</b></td>"
                else:
                    t += f"<td>{tmp_mean} ± {tmp_std}</td>"

            t += "\n</tr>\n"

    t += "</tbody>\n"
    t += "</table>\n"

    return t

with open("_template_PERFORMANCE_TABLES.md", "r") as f:
    text = f.read()

# Gather all data
statistics, significance = gatherData(".")

# Make the tables
for i, compute_table in enumerate([table1, table2, table3, table4, table5,
        table6, table7, table8, table9]):
    text = text.replace(f"[TABLE-{i+1}]", compute_table(statistics, significance))

# Saving the results
with open("PERFORMANCE_TABLES.md", "w") as f:
    f.write(text)
