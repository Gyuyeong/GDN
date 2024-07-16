from util.data import *
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch_geometric
from torch_geometric.data import Data
import networkx as nx

import torch


def get_full_err_scores(test_result, val_result):
    np_test_result = np.array(test_result)
    np_val_result = np.array(val_result)

    all_scores =  None
    all_normals = None
    feature_num = np_test_result.shape[-1]

    labels = np_test_result[2, :, 0].tolist()

    for i in range(feature_num):
        test_re_list = np_test_result[:2,:,i]
        val_re_list = np_val_result[:2,:,i]

        scores = get_err_scores(test_re_list, val_re_list)
        normal_dist = get_err_scores(val_re_list, val_re_list)

        if all_scores is None:
            all_scores = scores
            all_normals = normal_dist
        else:
            all_scores = np.vstack((
                all_scores,
                scores
            ))
            all_normals = np.vstack((
                all_normals,
                normal_dist
            ))

    return all_scores, all_normals


def get_final_err_scores(test_result, val_result):
    full_scores, all_normals = get_full_err_scores(test_result, val_result, return_normal_scores=True)

    all_scores = np.max(full_scores, axis=0)

    return all_scores

# plot TSNE on learned node embeddings
def get_tsne(embeddings:np.array, feature_map: list, dataset:str):
    tsne = TSNE(n_components=2, perplexity=5)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', marker='o')

    for i, label in enumerate(feature_map):
        plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), textcoords='offset points', xytext=(0, 10), ha='center')

    plt.title('T-SNE Visualization of Node Embeddings')
    plt.savefig(f'./results/{dataset}/tsne_node_embedding.png')
    plt.close()


# plot prediction, ground truth and anomalies based on the given threshold
def plot_anomalies(pred, gt, label, feature_map:list, dataset:str, test_error_score, thresholds):
    if isinstance(pred, list):
        pred = np.array(pred)
    if isinstance(gt, list):
        gt = np.array(gt)
    if isinstance(label, list):
        label = np.array(label)
    
    time_points = np.arange(pred.shape[0])

    anomaly = np.zeros(pred.shape, dtype=int)

    print("Plotting Anomalies ...")
    for sensor in tqdm(range(pred.shape[1])):
        sensor_pred = pred[:, sensor]
        sensor_gt = gt[:, sensor]
        sensor_label = label[:, sensor]

        plt.figure(figsize=(6, 4))
        plt.grid(True)
        plt.plot(time_points, sensor_pred, label="Prediction", c='black')
        plt.plot(time_points, sensor_gt, label="Ground Truth", c='green')

        for t in time_points:
            if sensor_label[t] == 1.0:
                plt.axvline(x=t, color='red', alpha=0.02)

            if test_error_score[t][sensor] > thresholds[sensor]:
                plt.plot(t, sensor_gt[t], 'ro', markersize=2)
                anomaly[t][sensor] = 1

        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.title(f"Sensor {feature_map[sensor]}\n# of Detected Anomalies: {np.sum(anomaly[:, sensor])} / {time_points.shape[0]}")
        plt.legend()
        plt.savefig(f"./results/{dataset}/anomaly_plots/{feature_map[sensor]}.png")
        plt.close()


def plot_graph_with_attention_weight(nodes, edge_index, edge_attr, feature_map, dataset):
    # normalize for opacity
    weights = edge_attr.cpu().detach().numpy()
    min_weights = weights.min(axis=1).reshape(-1, 1)
    max_weights = weights.max(axis=1).reshape(-1, 1)
    norm_weights = (weights - min_weights) / (max_weights - min_weights)
    norm_weights = norm_weights.flatten()

    graph = Data(x=nodes, edge_index=edge_index, edge_attr=norm_weights)
    g = torch_geometric.utils.to_networkx(graph, edge_attrs=['edge_attr'])

    labeldict = {idx: feature_map[idx] for idx in range(len(feature_map))}

    print("Plotting Graph ...")
    plt.figure(figsize=(12, 8))

    pos = nx.spring_layout(g)

    nx.draw_networkx_nodes(g, pos, node_size=300)

    for i, (u, v, d) in enumerate(g.edges(data=True)):
        opacity = norm_weights[i]
        nx.draw_networkx_edges(g, pos, edgelist=[(u, v)], alpha=opacity, edge_color='black')

    nx.draw_networkx_labels(g, pos, font_size=10, labels=labeldict)

    plt.title("Graph with Attention Weight")
    plt.savefig(f"./results/{dataset}/graph_attention_weight.png")
    plt.close()

def get_err_scores(test_res, val_res):
    test_predict, test_gt = test_res
    val_predict, val_gt = val_res

    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)

    test_delta = np.abs(np.subtract(
                        np.array(test_predict).astype(np.float64), 
                        np.array(test_gt).astype(np.float64)
                    ))
    epsilon=1e-2

    err_scores = (test_delta - n_err_mid) / ( np.abs(n_err_iqr) +epsilon)

    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i-before_num:i+1])

    
    return smoothed_err_scores


def get_individual_err_scores(predict: list, gt: list) -> np.array:
    pred_array = np.array(predict)
    true_array = np.array(gt)

    # error score
    delta = np.abs(pred_array - true_array)
    delta_t = delta.T

    # median and iqr
    median = np.median(delta_t, axis=1)
    q75, q25 = np.percentile(delta_t, [75, 25], axis=1)
    iqr = q75 - q25

    epsilon = 1e-4
    normalized_error_score = (delta - median) / (np.abs(iqr) + epsilon)

    # simple moving average for each tick t for each node
    smoothed_error_score = np.zeros(normalized_error_score.shape)
    before_num = 3
    for i in range(before_num, len(normalized_error_score)):
        smoothed_error_score[i] = np.mean(normalized_error_score[i-before_num: i+1], axis=0)

    return smoothed_error_score


def get_loss(predict, gt):
    return eval_mseloss(predict, gt)

def get_f1_scores(total_err_scores, gt_labels, topk=1):
    print('total_err_scores', total_err_scores.shape)
    # remove the highest and lowest score at each timestep
    total_features = total_err_scores.shape[0]

    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
    
    topk_indices = np.transpose(topk_indices)

    total_topk_err_scores = []
    topk_err_score_map=[]
    # topk_anomaly_sensors = []

    for i, indexs in enumerate(topk_indices):
       
        sum_score = sum( score for k, score in enumerate(sorted([total_err_scores[index, i] for j, index in enumerate(indexs)])) )

        total_topk_err_scores.append(sum_score)

    final_topk_fmeas = eval_scores(total_topk_err_scores, gt_labels, 400)

    return final_topk_fmeas

def get_val_performance_data(total_err_scores, normal_scores, gt_labels, topk=1):
    total_features = total_err_scores.shape[0]

    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]

    total_topk_err_scores = []
    topk_err_score_map=[]

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

    thresold = np.max(normal_scores)

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    f1 = f1_score(gt_labels, pred_labels)


    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return f1, pre, rec, auc_score, thresold


def get_best_performance_data(total_err_scores, gt_labels, topk=1):

    total_features = total_err_scores.shape[0]

    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]

    total_topk_err_scores = []
    topk_err_score_map=[]

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

    final_topk_fmeas ,thresolds = eval_scores(total_topk_err_scores, gt_labels, 400, return_thresold=True)

    th_i = final_topk_fmeas.index(max(final_topk_fmeas))
    thresold = thresolds[th_i]

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return max(final_topk_fmeas), pre, rec, auc_score, thresold

