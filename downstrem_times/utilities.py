import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math
import random
from loggers import logger_f#metricslogger
from itertools import combinations
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib

def transform_dict_to_lists(dic,clients):
    """This function take a dic that contains the evluation of each coalion
    And output this to the format of binary tuples and acc"""

    sets = []
    accuracies = []

    for key, value in dic.items():
        binary_list = [0] * clients
        for index in key:
            binary_list[index] = 1
        sets.append(binary_list)
        accuracies.append(value)

    return sets, accuracies


def shapley(clients, groups, acc):
    '''
    compute the Shapley Value
        (input)  clients: client number
        (input)  groups:  coalitions with binary assignment matrix, e.g. for 2 players [[0,0],[1,0],[0,1],[1,1]]
        (input)  acc:     accuracies of the corresponding groups, e.g., for two players [a, b, c, d]
        (output) score:   Shapley value of the players
    '''
    scores = np.zeros(clients)
    for i in range(clients):
        tmp = 0
        for j, subset in enumerate(groups):
            if subset[i] == 0:
                continue
            subset_without_i = np.copy(subset)
            subset_without_i[i] = 0
            subset_index = j
            idx = [k for k, l in enumerate(groups) if all(x == y for x, y in zip(l, subset_without_i))]
            subset_without_i_index = idx[0]
            marginal_contribution = acc[subset_index] - acc[subset_without_i_index]
            subset_size = np.sum(subset) - 1
            weight = (math.factorial(subset_size) * math.factorial(clients - subset_size - 1)) / math.factorial(clients)
            tmp += weight * marginal_contribution
        scores[i] = tmp
    return scores.tolist()


def cosine_similarity_models(model_client,model_server):
    """
    Compute the cosine similarity between the parameters of two PyTorch models.
    """
    params_a = torch.cat([p.view(-1) for p in model_client.parameters()])
    params_b = torch.cat([p.view(-1) for p in model_server.parameters()])
    
    similarity = F.cosine_similarity(params_a.unsqueeze(0), params_b.unsqueeze(0))
    
    return similarity.item()


def banzhaf(clients, groups, acc):
    '''
    compute the Shapley Value
        (input)  clients: client number
        (input)  groups:  coalitions with binary assignment matrix, e.g. for 2 players [[0,0],[1,0],[0,1],[1,1]]
        (input)  acc:     accuracies of the corresponding groups, e.g., for two players [a, b, c, d]
        (output) score:   Shapley value of the players
    '''
    scores = np.zeros(clients)
    for i in range(clients):
        tmp = 0
        for j, subset in enumerate(groups):
            if subset[i] == 0:
                continue
            subset_without_i = np.copy(subset)
            subset_without_i[i] = 0
            subset_index = j
            idx = [k for k, l in enumerate(groups) if all(x == y for x, y in zip(l, subset_without_i))]
            subset_without_i_index = idx[0]
            marginal_contribution = acc[subset_index] - acc[subset_without_i_index]
            weight = (1/2**(clients-1))
            tmp += weight * marginal_contribution
        scores[i] = tmp
    return scores.tolist()

def pri_ce(clients, groups, acc):
    i1i = np.zeros(clients)
    l1o = np.zeros(clients)
    ieei = np.zeros(clients)
    leeo = np.zeros(clients)
    include = np.zeros(clients)
    leave = np.zeros(clients)
    for i, subset in enumerate(groups):
        if np.sum(subset) == 0:
            null = acc[i]
        if np.sum(subset) == clients:
              grand = acc[i]
        if np.sum(subset) == 1:
            tmp = [j for j, k in enumerate(subset) if k == 1]
            include[tmp[0]] = acc[i]
        if np.sum(subset) == clients - 1:
            tmp = [j for j, k in enumerate(subset) if k == 0]
            leave[tmp[0]] = acc[i]
    for i in range(clients):
        i1i[i] = include[i] - null
        l1o[i] = grand - leave[i]
        for j in range(clients):
            ieei[i] += (leave[j] - null)
            leeo[i] += (grand - include[j])
        ieei[i] -= (leave[i] - null)
        leeo[i] -= (grand - include[i])
    ieei= ieei/(clients - 1) ** 2
    leeo= leeo/(clients - 1) ** 2
    se = (i1i + l1o)/2
    ee = (ieei + leeo)/2
    # beta_ee = np.sum(ee)
    beta_se = np.sum(ee)
    fp = se*(grand/beta_se)
    # bee = ee*(grand/beta_ee)
    # logger_f(f"bee: {bee.tolist()}",f"{path}/values/ppce_global.log")
    return l1o.tolist(),fp.tolist(),ee.tolist()#,bee.tolist()


def affine_trans_list(lst):
    numbers=np.array(lst)
    if np.min(numbers)<0:
        tmp=numbers - np.min(numbers)
        tmp_mean=np.mean(tmp)
        trans=tmp/tmp_mean
    elif np.min(numbers)==0 and np.max(numbers)==0:
        trans=lst
    else:
        pos_mean=np.mean(lst)
        trans=lst/pos_mean
    return trans

def affine_trans(ppce):
    tmp=[]
    for list_ in ppce:
        tmp.append(affine_trans_list(list_))
    return tmp

def barplot_from_list(
    data,
    save_path=None,
    title="Box Plot",
    show=False
):

    x=range(len(data))
    labels=[f"Cl {i}" for i in range(len(data))]
    plt.xticks(x, labels)
    ylabel="Values",
    plt.figure()
    plt.bar(x,data)
    plt.title(title)
    plt.ylabel(ylabel)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    if show:
        plt.show()
    else:
        plt.close()

    
def plot_losses(iterations, losses_mean, losses_var=None, save_path="./plot_of_losses/losses_weight_score_negative_cifar.png"):
    """Plot multiple loss curves with optional shaded bands using variances.

    Args:
        iterations (int): Number of iterations (x-axis length).
        losses_mean (sequence): Sequence (len 5) of mean loss lists/arrays (shape: [5, iterations]).
        losses_var (sequence, optional): Sequence (len 5) of variance lists/arrays (shape: [5, iterations]). If provided, will plot shaded bands mean +/- var.
        save_path (str): Path to save the resulting plot.
    """
    plt.figure(figsize=(8, 5))
    plt.title('Plot of performances')
    plt.xlabel('Global iteration')
    plt.ylabel('Negative average loss')

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    labels = ["wg sv", "wg banz", "wg l1o", "wg fp", "wg ee", "FedAve"] #"wg banz"

    for i, label in enumerate(labels):
        mean = np.array(losses_mean[i])
        color = colors[i % len(colors)]
        plt.plot(range(iterations), mean, label=label, color=color)

        if losses_var is not None:
            try:
                var = np.array(losses_var[i])
                lower = mean - var
                upper = mean + var
                plt.fill_between(range(iterations), lower, upper, color=color, alpha=0.2)
            except Exception:
                # If shapes don't match or missing entry, skip shading for this line
                pass

    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add a light grid for readability
    # Ensure output directory exists
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    #plt.show()




if __name__ == "__main__":

    data = [1, 2, 2, 3, 4, 5, 10]

    barplot_from_list(
        data,
        save_path="./plots_cl_scores/single_boxplot.png",
        title="Single Dataset"
    )



