import random
import torch
import torch.nn as nn
import numpy as np
import copy
import os
import time
import matplotlib.pyplot as plt
from loggers import logger_f
from modelos import CNN, MLP, resnet
from tqdm import tqdm 
from data_partition import cl_data_load_and_testset
#from data_partition_isic import cl_data_load_and_testset_isic
from itertools import combinations
from utilities import shapley, pri_ce, banzhaf, cosine_similarity_models, transform_dict_to_lists, affine_trans, barplot_from_list, plot_losses


# random_seed = 42
# random.seed(random_seed)
# np.random.seed(random_seed)
# torch.manual_seed(random_seed)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#print(f"Using device: {device}")


class Client:
    def __init__(self, model_type, data_loader, device=device):
        self.device = device
        self.data_loader = data_loader
        self.model_type = model_type
        self.model = self.init_model()
        self.client_size = len(self.data_loader.dataset) #+ len(self.testing_cli.dataset)
        
    def init_model(self):
        if self.model_type == "MLP":
            return MLP()
        elif self.model_type == "CNN":
            return CNN()
        elif self.model_type == "resnet":
            return resnet()
        else:
            raise ValueError("Unsupported model type")

    def fit(self, epochs=3, desc="Client"):
        # Training loop
        criterion = nn.CrossEntropyLoss()
        self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)#, weight_decay=0.01)
        losses = []
        # Standard training loop with progress bar
        for epoch in tqdm(range(epochs), desc=desc, leave=False):
            epoch_loss = 0
            for x, y in self.data_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(self.data_loader)
            losses.append(avg_loss)

        return losses


    def evaluate_n_loss(self, test_data):
        """
        Evaluates model using negative loss (higher is better, like accuracy)
        Args:
            model: neural network model
            test_data: test data loader
            device: computing device
        Returns:
            float: negative average loss
        """
        self.model.to(self.device)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for x, y in test_data:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = criterion(outputs, y)
                total_loss += loss.item()
                num_batches += 1
                
        avg_loss = total_loss / num_batches
        # Return negative loss so higher values are better (like accuracy)
        return -avg_loss
    

    def reset_parameters(self):
        # self.normalizer.reset_parameters()
        # self.layers.reset_parameters()
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def set_parameters(self,params):
        for model_parametro, param in zip(self.model.parameters(), params):
            model_parametro.data = param

    def load_state_dict(self, weights):
        self.model.load_state_dict(weights)

    def parameters(self):
        return self.model.parameters()

class Federation:
    def __init__(self, model_type, num_clients, data, device=device):
        self.device =device
        self.num_clients=num_clients
        self.model_type=model_type
        self.data = data
        self.model = self.init_model().to(self.device)
        self.data_clients = self.data[0]
        self.test = self.data[1]
        self.lista_clientes=[Client(model_type,self.data_clients[i]["train_loader"]) for i in range(self.num_clients)]


    def init_model(self):
        if self.model_type == "MLP":
            return MLP()
        elif self.model_type == "CNN":
            return CNN()
        elif self.model_type == "resnet":
            return resnet()
        else:
            raise ValueError("Unsupported model type")


    def total_size(self):
        size = 0
        for client in self.lista_clientes:
            size += client.client_size
        return size


    def aggregate(self, List_clients, weights=None):
        """
        Aggregate client models. If `weights` is provided it should be a sequence
        with one weight per client (will be normalized). If `weights` is None,
        a simple average is used.
        """
        pesos_cliente = [list(client.parameters()) for client in List_clients]
        num_clients = len(List_clients)
        if weights is None:
            weights = [1.0 / num_clients] * num_clients
        else:
            weights = np.array(weights, dtype=float)
            if weights.sum() == 0:
                raise ValueError("Sum of weights must be > 0")
            weights = (weights / weights.sum()).tolist()
        data = []
        for params_clients in zip(*pesos_cliente):
            # params_clients: tuple of nn.Parameter for each client for a given param
            weighted = sum(w * p.data.to(self.device) for p, w in zip(params_clients, weights))
            data.append(weighted)
        return data
    
 
    def federated_averaging(self, combo, weights=None, num_iterations=5, num_epochs=3, desc="FedRound"):
        conj_clientes = [self.lista_clientes[i] for i in combo]
        losses = []
        
        for iteration in tqdm(range(num_iterations), desc=desc, leave=False):
            # Local training phase: each client trains for num_epochs
            for client_idx, client in enumerate(conj_clientes):
                weights_modelo = self.model.state_dict()
                client.load_state_dict(weights_modelo)
                client.fit(num_epochs, desc=f"  Client {client_idx} epochs")
            
            # Aggregation phase
            params = self.aggregate(conj_clientes, weights)
            self.set_parameters(params)
            loss = self.evaluate_n_loss()
            losses.append(loss)
        
        return losses 
        
    def evaluate_n_loss(self):
            """
            Evaluates model using negative loss (higher is better, like accuracy)
            Args:
                model: neural network model
                test_data: test data loader
                device: computing device
            Returns:
                float: negative average loss
            """
            self.model.to(self.device)
            self.model.eval()
            criterion = nn.CrossEntropyLoss()
            total_loss = 0
            num_batches = 0
            
            with torch.no_grad():
                for x, y in self.test:
                    x, y = x.to(self.device), y.to(self.device)
                    outputs = self.model(x)
                    loss = criterion(outputs, y)
                    total_loss += loss.item()
                    num_batches += 1
                    
            avg_loss = total_loss / num_batches
            # Return negative loss so higher values are better (like accuracy)
            return -avg_loss


    def set_parameters(self,params):
        for model_parametro, param in zip(self.model.parameters(), params):
            model_parametro.data = param

    def reset_parameters(self):
        # self.normalizer.reset_parameters()
        # self.layers.reset_parameters()
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def parameters(self):
        self.model.parameters()

class Scores(Federation):
    def __init__(self, model_type, num_clients, data, device=device):
        super().__init__(model_type, num_clients, data, device=device)

    def coal_value_at_round(self, iterations=3, weights=None, logger_path="./", epochs=3):
        """
        Compute coalitional values for Shapley value and privacy measures.
        """
        losses = self.federated_averaging(list(range(self.num_clients)), weights, iterations, epochs, desc="CoalVal:FedAvg")
        d = dict()
        d[()] = 0
        timing = {
            "client_singleton_test_eval_seconds": {},
            "coalition_test_eval_seconds": {},
            "coalition_aggregation_seconds": {}
        }
        
        # Single-client coalitions
        for client in tqdm(range(self.num_clients), desc="CoalVal:Singles", leave=False):
            eval_start = time.perf_counter()
            d[tuple([client])] = self.lista_clientes[client].evaluate_n_loss(self.test)
            elapsed = time.perf_counter() - eval_start
            coalition_key = str(tuple([client]))
            timing["client_singleton_test_eval_seconds"][f"client_{client}"] = elapsed
            timing["coalition_test_eval_seconds"][coalition_key] = elapsed
        
        # Multi-client coalitions
        for subset_len in tqdm(range(2, self.num_clients + 1), desc="CoalVal:Subsets", leave=False):
            subset_count = len(list(combinations(list(range(self.num_clients)), subset_len)))
            for subset_idx in tqdm(combinations(list(range(self.num_clients)), subset_len), 
                                    total=subset_count, desc=f"  Size {subset_len}", leave=False):
                part = [self.lista_clientes[i] for i in subset_idx]
                if weights is None:
                    subset_wgs = None
                else:
                    subset_wgs = [weights[i] for i in subset_idx]
                aggregation_start = time.perf_counter()
                params = self.aggregate(part, subset_wgs)
                timing["coalition_aggregation_seconds"][str(subset_idx)] = time.perf_counter() - aggregation_start
                self.set_parameters(params)
                eval_start = time.perf_counter()
                d[subset_idx] = self.evaluate_n_loss()
                timing["coalition_test_eval_seconds"][str(subset_idx)] = time.perf_counter() - eval_start
        
        logger_f(f"at coalitional values: {d}", f"{logger_path}values/results_coalitons.log")
        logger_f(f"coalition_timing_seconds: {timing}", f"{logger_path}values/timings.log")
        return d, losses
    
    def cosine_sim_clie_server(self):
        co_si=[]
        for client in self.lista_clientes:
            value=cosine_similarity_models(client.model,self.model)
            co_si.append(value)
        logger_f(f"Cosine: {co_si}","cosine")
        return co_si


    def compute_weights(self, iterations=5, epochs=3, path="./"):
        no_wg, losses = self.coal_value_at_round(iterations=iterations,epochs=epochs)
        group, acc = transform_dict_to_lists(no_wg,self.num_clients)
        privacy_start = time.perf_counter()
        privacy = pri_ce(self.num_clients ,group, acc)
        privacy_seconds = time.perf_counter() - privacy_start
        sv_start = time.perf_counter()
        sv = shapley(self.num_clients, group, acc)
        sv_seconds = time.perf_counter() - sv_start
        #banz = banzhaf(self.num_clients, group, acc)
        cos_start = time.perf_counter()
        cos = self.cosine_sim_clie_server()
        cos_seconds = time.perf_counter() - cos_start
        scores = [sv, cos, privacy[0], privacy[1], privacy[2]]
        affine_start = time.perf_counter()
        aff_scores = affine_trans(scores)
        affine_seconds = time.perf_counter() - affine_start
        server_timing = {
            "privacy_l10_fp_ee_seconds": privacy_seconds,
            "SV_seconds": sv_seconds,
            "COS_seconds": cos_seconds,
            "affine_transform_all_scores_seconds": affine_seconds
        }
        logger_f(f"server_metric_timing_seconds: {server_timing}", f"{path}values/timings.log")
        logger_f(f"SV: {aff_scores[0]}",f"{path}values/shapleys.log")
        logger_f(f"COS: {aff_scores[1]}",f"{path}values/cosine.log")
        logger_f(f"l10: {aff_scores[2]}",f"{path}values/ppce_global.log")
        logger_f(f"fp: {aff_scores[3]}",f"{path}values/ppce_global.log")
        logger_f(f"ee: {aff_scores[4]}",f"{path}values/ppce_global.log")
        return aff_scores, losses
    
    def last_client_is_lowest(self, aff_scores):
        """
        Check if the last client has the lowest score for each score metric.
        
        Args:
            aff_scores: List of score arrays, where each array contains per-client scores
        
        Returns:
            List of 1s and 0s, where 1 indicates the last client has the lowest score,
            0 indicates otherwise
        """
        result = []
        for score in aff_scores:
            # Find the index of the minimum value
            min_idx = np.argmin(score)
            # Check if the last client (index -1) has the lowest score
            last_client_idx = len(score) - 1
            is_lowest = 1 if min_idx == last_client_idx else 0
            result.append(is_lowest)
        return result
    
    def compute_and_score_lowest(self, rounds=5, epochs=3):
        """
        Compute weights over multiple rounds and score how often the last client
        ends up with the lowest score for each metric.
        
        Args:
            rounds: Number of rounds to run compute_weights
            iterations: Number of iterations for federated averaging
            epochs: Number of epochs for local training
        
        Returns:
            List of 5 scores (between 0 and 1) representing the proportion of times
            the last client had the lowest score for each metric (SV, cos, l10, fp, ee)
        """
        lowest_counts = [0, 0, 0, 0, 0]  # Track counts for each of 5 metrics
        
        for round_idx in tqdm(range(rounds), desc="Computing scores", leave=False):
            aff_scores, _ = self.compute_weights(iterations=round_idx, epochs=epochs)
            is_lowest_list = self.last_client_is_lowest(aff_scores)
            
            # Accumulate counts
            for metric_idx, is_lowest in enumerate(is_lowest_list):
                lowest_counts[metric_idx] += is_lowest
            
            # Reset parameters for next round
            self.reset_parameters()
            for cl in range(self.num_clients):
                self.lista_clientes[cl].reset_parameters()
        
        # Compute scores as proportions (0 to 1)
        score_results = [count / rounds for count in lowest_counts]
        return score_results
    
    def lowest_score_mean_var(self, global_iterations=5, rounds=5, epochs=3):
        """
        Compute mean and variance of lowest scores across multiple global iterations.
        
        Args:
            global_iterations: Number of global iterations to run
            rounds: Number of rounds for each compute_and_score_lowest call
            epochs: Number of epochs for local training
        
        Returns:
            Dictionary with keys:
            - "scores_mean": Array of 5 mean values (one for each metric)
            - "scores_var": Array of 5 variance values (one for each metric)
            - "scores_list": List of all individual scores for plotting
        """
        scores_list = []
        
        for gl_iter in tqdm(range(global_iterations), desc="GlobalIter", leave=True):
            scores = self.compute_and_score_lowest(rounds=rounds, epochs=epochs)
            scores_list.append(scores)
            
            # Reset parameters for next global iteration
            self.reset_parameters()
            for cl in range(self.num_clients):
                self.lista_clientes[cl].reset_parameters()
        
        # Stack into array
        # Shape: (global_iterations, 5)
        scores_tensor = np.stack(scores_list, axis=0)
        
        # Compute mean and variance across global iterations
        scores_mean = np.mean(scores_tensor, axis=0)
        scores_var = np.var(scores_tensor, axis=0)

        d = {
            "scores_mean": scores_mean,
            "scores_var": scores_var,
            "scores_list": scores_list
        }
        
        logger_f(f"Statistics attack score: {d}", "./values/values_attack_score.log")

        return d
    
    def plot_lowest_scores_boxplot(self, global_iterations=5, rounds=5, epochs=3, save_path="./plots_cl_scores/boxplot_lowest_scores.png"):
        """
        Create a box plot showing the distribution of lowest scores for each metric.
        
        Args:
            global_iterations: Number of global iterations to run
            rounds: Number of rounds for each compute_and_score_lowest call
            epochs: Number of epochs for local training
            save_path: Path to save the plot image
        """
        # Compute scores
        results = self.lowest_score_mean_var(global_iterations=global_iterations, rounds=rounds, epochs=epochs)
        scores_list = results["scores_list"]
        
        # Prepare data for box plot
        # scores_list is a list of lists, convert to numpy array and transpose to get per-metric data
        scores_array = np.array(scores_list)  # Shape: (global_iterations, 5)
        
        # Create box plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = ["SV", "COS", "l10", "fp", "ee"]
        box_data = [scores_array[:, i] for i in range(5)]
        
        bp = ax.boxplot(box_data, labels=metric_names, patch_artist=True)
        
        # Customize colors
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax.set_xlabel("Metrics", fontsize=12)
        ax.set_ylabel("Score (0-1)", fontsize=12)
        ax.set_title("Distribution of Lowest Scores by Metric", fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger_f(f"Box plot saved to {save_path}", "./values/plot.log")
    
    def scores_losses(self, iterations=5, epochs=3):
        list_scores, losses_base = self.compute_weights(iterations=iterations, epochs=epochs)
        self.reset_parameters()
        for cl in range(self.num_clients):
            self.lista_clientes[cl].reset_parameters()

        losses_plots = []
        for score_idx, wg in enumerate(tqdm(list_scores, desc="  ScoreWeights", leave=False)):
            los = self.federated_averaging(list(range(self.num_clients)), wg, iterations, epochs, 
                                          desc=f"    Score{score_idx} rounds")
            losses_plots.append(los)
            self.reset_parameters()
            for cl in range(self.num_clients):
                self.lista_clientes[cl].reset_parameters()
        losses_plots.append(losses_base)
        return np.array(list_scores), np.array(losses_plots)
        
    def losses_mean_var(self, gl_iteration=5, fed_iteration=5, epochs=3):
        scores_list = []
        losses_list = []

        for gl_iter in tqdm(range(gl_iteration), desc="GlobalIter", leave=True):
            scores, losses = self.scores_losses(
                iterations=fed_iteration,
                epochs=epochs
            )
            scores_list.append(scores)
            losses_list.append(losses)

        # Stack into tensors
        # Shape: (gl_iteration, ..., ...)
        scores_tensor = np.stack(scores_list, axis=0)
        losses_tensor = np.stack(losses_list, axis=0)

        # Mean and variance across global iterations
        scores_mean = np.mean(scores_tensor, axis=0)
        scores_var  = np.var(scores_tensor, axis=0)

        losses_mean = np.mean(losses_tensor, axis=0)
        losses_var  = np.var(losses_tensor, axis=0)

        return {
            "scores_mean": scores_mean,
            "scores_var": scores_var,
            "losses_mean": losses_mean,
            "losses_var": losses_var
        }
            
    def plots_losses_scores(self, gl_ite=3, iterations=5, epochs=3):
        # compute means over global iterations
        dict_sc_lo = self.losses_mean_var(gl_iteration=gl_ite, fed_iteration=iterations, epochs=epochs)

        scores_arr = dict_sc_lo["scores_mean"]
        losses_arr = dict_sc_lo["losses_mean"]
        losses_var = dict_sc_lo["losses_var"]
        scores_var = dict_sc_lo["scores_var"]
         
        

        # ensure output directories exist
        os.makedirs("./plots_cl_scores", exist_ok=True)
        os.makedirs("./plot_of_losses", exist_ok=True)
        os.makedirs("./values", exist_ok=True)

        # plot and log scores (bar plots)
        scores_names = ["sv", "COS", "l10", "fp", "ee"]
        for sc, sc_name in zip(scores_arr, scores_names):
            save_path = f"./plots_cl_scores/plot_mean_{sc_name}.png"
            barplot_from_list(sc, save_path=save_path, title=f"Plot of {sc_name}")
            logger_f(f"scores_mean_{sc_name}: {sc.tolist()}", f"./values/scores_{sc_name}.log")

        # log the whole arrays for convenience
        logger_f(f"scores_mean_all: {scores_arr.tolist()}", f"./values/scores_mean_all.log")
        logger_f(f"losses_mean_all: {losses_arr.tolist()}", f"./values/losses_mean_all.log")
        logger_f(f"losses_var_all: {losses_var.tolist()}", f"./values/losses_var_all.log")
        logger_f(f"scores_var_all: {scores_var.tolist()}", f"./values/scores_var_all.log")

        # still write per-loss logs (no individual plots)
        losses_names = ["sv", "COS", "l10", "fp", "ee", "FedAvg"]
        for idx, name in enumerate(losses_names):
            try:
                logger_f(f"losses_{name}: {losses_arr[idx].tolist()}", f"./values/losses_{name}.log")
            except Exception:
                logger_f(f"losses_{name}: n/a", f"./values/losses_{name}.log")

        # single combined plot for all losses
        try:
            iter_len = losses_arr.shape[1]
        except Exception:
            iter_len = iterations
        # pass both mean and variance so plot shows shaded bands
        plot_losses(iter_len, losses_arr, dict_sc_lo.get("losses_var", None))

        return None


if __name__ == "__main__":
    num_clients = 3
    model_type = "CNN"
    data = cl_data_load_and_testset(dataname="cifar10", num_clients=3, noise_clients={0:0.0,1:0.25,2:0.5})
    #noise = {0: 0.0, 1: 0.1, 2: 0.3}
    #data = cl_data_load_and_testset_isic(num_clients=3, noise_clients=noise, test_size=0.2)    
    data_clientes = data[0]
    test_data = data[1]
    score = Scores(model_type, num_clients, data)
    score.plots_losses_scores(gl_ite=5,iterations=10,epochs=3)

    #score.plot_lowest_scores_boxplot(global_iterations=10, rounds=10, epochs=1,save_path="./plots_cl_scores/boxplot_lowest_scores.png")
    #print("Box plot saved to ./plots_cl_scores/boxplot_lowest_scores.png")
    
    # coalitions = score.coal_value_at_round()
    # group, acc = transform_dict_to_lists(coalitions,num_clients)
    # privacy = pri_ce(num_clients ,group, acc)
    # print(shapley(num_clients,group, acc))
    # print(privacy[0])
    # print(privacy[1])
    # print(privacy[2])
    # print(privacy[3])
    # print(coalitions)

    # federation = Federation(model_type, num_clients, data)
    # trains = federation.federated_averaging([0,1,2], weights)
    # losses=trains
    # plt.plot(range(global_rounds), losses)
    # plt.show()
