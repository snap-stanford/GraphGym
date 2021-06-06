"""
Metrics, other utility, and helper functions.
"""
import deepsnap
import numpy as np
import torch
from graphgym.config import cfg
from graphgym.loss import compute_loss
from torch_scatter import scatter_max, scatter_mean, scatter_min
# TODO: proof-read this file.
# TODO: remove comments.


def get_keep_ratio(existing: torch.Tensor,
                   new: torch.Tensor,
                   mode: str = 'linear') -> torch.Tensor:
    """
    Get the keep ratio for individual nodes to update node embeddings.
    Specifically:
       state[v,t] = state[v,t-1]*keep_ratio + new_feature[v,t]*(1-keep_ratio)

    Args:
        existing: a tensor of nodes' degrees in G[0], G[1], ..., G[t-1].
        new: a tensor of nodes' degrees in G[t].
        mode: how to compute the keep_ratio.

    Returns:
        A tensor with shape (num_nodes,) valued in [0, 1].
    """
    if mode == 'constant':
        # This scheme is equivalent to exponential decaying.
        ratio = torch.ones_like(existing)
        # node observed for the first time, keep_ratio = 0.
        ratio[torch.logical_and(existing == 0, new > 0)] = 0
        # take convex combination of old and new embeddings.
        # 1/2 can be changed to other values.
        ratio[torch.logical_and(existing > 0, new > 0)] = 1 / 2
        # inactive nodes have keep ratio 1, embeddings don't change.
    elif mode == 'linear':
        # The original method proposed by Jiaxuan.
        ratio = existing / (existing + new + 1e-6)
    # Following methods aim to shrink the weight of existing
    # degrees, help to ensure non-trivial embedding update when the graph
    # is large and history is long.
    elif mode == 'log':
        ratio = torch.log(existing + 1) / (
            torch.log(existing + 1) + new + 1e-6)
    elif mode == 'sqrt':
        ratio = torch.sqrt(existing) / (torch.sqrt(existing) + new + 1e-6)
    else:
        raise NotImplementedError(f'Mode {mode} is not supported.')
    return ratio


def size_of(batch: deepsnap.graph.Graph) -> int:
    """Computes how much memory a batch has consumed."""
    total_byte = 0
    for k, v in batch.__dict__.items():
        if isinstance(v, torch.Tensor):
            total_byte += v.element_size() * v.nelement()
        elif isinstance(v, list):  # for node_states.
            for sub_v in v:
                if isinstance(sub_v, torch.Tensor):
                    total_byte += sub_v.element_size() * sub_v.nelement()

    return total_byte / (1024 ** 2)  # MiB.


def move_batch_to_device(batch: deepsnap.graph.Graph,
                         device: str) -> deepsnap.graph.Graph:
    """Moves and collects everything in the batch to the target device."""
    device = torch.device(device)
    # This handles node_feature, edge_feature, etc.
    batch = batch.to(device)

    for layer in range(len(batch.node_states)):
        if torch.is_tensor(batch.node_states[layer]):
            batch.node_states[layer] = batch.node_states[layer].to(device)

    if hasattr(batch, 'node_cells'):
        # node_cells exist only for LSTM type RNNs.
        for layer in range(len(batch.node_cells)):
            if torch.is_tensor(batch.node_cells[layer]):
                batch.node_cells[layer] = batch.node_cells[layer].to(device)

    return batch


def edge_index_difference(edge_include: torch.LongTensor,
                          edge_except: torch.LongTensor,
                          num_nodes: int) -> torch.LongTensor:
    """Set difference operator, return edges in edge_all but not
        in edge_except.

    Args:
        edge_all (torch.LongTensor): (2, E1) tensor of edge indices.
        edge_except (torch.LongTensor): (2, E2) tensor of edge indices to be
            excluded from edge_all.
        num_nodes (int): total number of nodes.

    Returns:
        torch.LongTensor: Edge indices in edge_include but not in edge_except. 
    """
    # flatten (i, j) edge representations.
    idx_include = edge_include[0] * num_nodes + edge_include[1]
    idx_except = edge_except[0] * num_nodes + edge_except[1]
    # filter out edges in idx_except.
    mask = torch.from_numpy(np.isin(idx_include, idx_except)).to(torch.bool)
    idx_kept = idx_include[~mask]
    i = idx_kept // num_nodes
    j = idx_kept % num_nodes
    return torch.stack([i, j], dim=0).long()


def gen_negative_edges(edge_index: torch.LongTensor,
                       num_neg_per_node: int,
                       num_nodes: int) -> torch.LongTensor:
    """Generates a fixed number of negative edges for each node.

    Args:
        edge_index (torch.LongTensor): (2, E) array of positive edges.
        num_neg_per_node (int): 'approximate' number of negative edges generated
            for each source node in edge_index.
        num_nodes (int): total number of nodes.

    Returns:
        torch.LongTensor: approximate num_nodes * num_neg_per_node
            negative edges.
    """
    src_lst = torch.unique(edge_index[0])  # get unique senders.
    num_neg_per_node = int(1.2 * num_neg_per_node)  # add some redundancy.
    i = src_lst.repeat_interleave(num_neg_per_node)
    j = torch.Tensor(np.random.choice(num_nodes, len(i), replace=True))
    # candidates for negative edges, X candidates from each src.
    candidates = torch.stack([i, j], dim=0).long()
    # filter out positive edges in candidate.
    neg_edge_index = edge_index_difference(candidates, edge_index.to('cpu'),
                                           num_nodes)
    return neg_edge_index


# def compute_src_mrr_and_recall(edge_label_index: torch.LongTensor,
#                                edge_label: torch.LongTensor,
#                                pred_score: torch.Tensor,
#                                recall_k_lst: List[int],
#                                mrr_top_k: Optional[int] = None
#                                ) -> (float, Dict[int, float]):
#     """
#     Computes source-based MRR and recall at K for each source node in
#         edge_label_index.

#     Args:
#         edge_label_index: combination of positive and negative edges.
#         edge_label: label of edges in edge_label_index.
#         pred_score: P(E=positive) for each edge in edge_label_index.
#         recall_k_lst: to report recall at k for all k in this list.
#         mrr_top_k: calculating MRR for each source node using mean(1/rank) for
#             k positive edges with the highest pred_score. Set to None to use
#             all positive edges.
#     """
#     assert edge_label_index.shape[1] == len(edge_label) == len(pred_score)

#     src_lst = torch.unique(edge_label_index[0])  # source nodes to consider.
#     # edge_label_index were constructed by adding negative edges to every
#     # node in edge_index[0], thus every node in src_lst has at least one
#     # positive edge in edge_label_index.
#     # I.e., src_lst == torch.unique(edge_label_index[0][edge_label == 1])

#     node_level_mrr = []  # store MRR for each node.
#     node_recall_at = dict((k, []) for k in recall_k_lst)
#     for src in tqdm(src_lst, leave=False, desc='Node level MRR/Recall'):
#         # get positive/negative edges emitted from src node.
#         self_mask = (edge_label_index[0] == src)
#         self_label = edge_label[self_mask]
#         self_pred_score = pred_score[self_mask]

#         # Alternative implementation.
#         best = torch.max(self_pred_score[self_label == 1])
#         rank = torch.sum(self_pred_score[self_label == 0] >= best) + 1
#         # print(pos_edge_rank[0], true, torch.sum(label == 0))
#         mrr = float(1 / rank)
#         node_level_mrr.append(mrr)  # mrr for this node.

#         for k in recall_k_lst:
#             recall = _calculate_recall_at_k(self_pred_score, self_label, k)
#             node_recall_at[k].append(recall)

#     # Average over all nodes.
#     macro_recall = dict((k, np.mean(v)) for (k, v) in node_recall_at.items())
#     macro_mrr = float(np.mean(node_level_mrr))
#     return macro_mrr, macro_recall


# def _calculate_recall_at_k(pred_score: torch.Tensor,
#                            label: torch.Tensor,
#                            k: int) -> int:
#     """Computes whether the score of the most confident positive edge is
#         within the highest k scores. I.e., whether the most confident
#         positive edge beats at least k most confident negative edges.

#     Args:
#         pred_score: a tensor of scores of predictions.
#         label: a tensor of labels.
#         k: get whether successful recall at k.

#     Returns:
#         an indicator whether there is a successful recall at rank k.
#     """
#     neg_score = pred_score[label == 0]
#     if len(neg_score) == 0:
#         return 0
#     best_pos_score = torch.max(pred_score[label == 1])
#     rank = torch.sum(neg_score >= best_pos_score) + 1
#     return int(rank <= k)


@torch.no_grad()
def fast_batch_mrr(edge_label_index: torch.Tensor,
                   edge_label: torch.Tensor,
                   pred_score: torch.Tensor,
                   num_neg_per_node: int,
                   num_nodes: int,
                   method: str) -> float:
    """
    A vectorized implementation to compute average rank-based metrics over
        all source nodes.

    Args:
        edge_label_index:
        edge_label:
        pred_score: P(edge i is positive) from the model.
        num_neg_per_node: number of negative edges per node.
        num_nodes: total number of nodes in the graph.

    Returns:
        the MRR for all nodes.
    """
    # A list of source nodes to consider.
    src_lst = torch.unique(edge_label_index[0], sorted=True)
    num_users = len(src_lst)

    edge_pos = edge_label_index[:, edge_label == 1]
    edge_neg = edge_label_index[:, edge_label == 0]

    # By construction, negative edge index should be sorted by their src nodes.
    assert torch.all(edge_neg[0].sort()[0] == edge_neg[0])

    # Prediction scores of all positive and negative edges.
    p_pos = pred_score[edge_label == 1]
    p_neg = pred_score[edge_label == 0]

    # For each player src, compute the highest score among all positive edges
    # from src.
    # We want to compute the rank of this edge.
    # Construct an interval of model's performance.
    if method == 'mean':
        best_p_pos = scatter_mean(src=p_pos, index=edge_pos[0],
                                  dim_size=num_nodes)
    elif method == 'min':
        best_p_pos, _ = scatter_min(src=p_pos, index=edge_pos[0],
                                    dim_size=num_nodes)
    elif method == 'max':
        # The default setting, consider the rank of the most confident edge.
        best_p_pos, _ = scatter_max(src=p_pos, index=edge_pos[0],
                                    dim_size=num_nodes)
    else:
        raise ValueError(f'Unrecognized method: {method}.')
    # best_p_pos has shape (num_nodes), for nodes not in src_lst has value 0.
    best_p_pos_by_user = best_p_pos[src_lst]

    uni, counts = torch.unique(edge_neg[0], sorted=True, return_counts=True)
    # note: edge_neg (src, dst) are sorted by src.
    # find index of first occurrence of each src in edge_neg[0].
    # neg edges[0], [1,1,...1, 2, 2, ... 2, 3, ..]
    first_occ_idx = torch.cumsum(counts, dim=0) - counts
    add = torch.arange(num_neg_per_node, device=first_occ_idx.device)

    # take the first 100 negative edges from each src.
    score_idx = first_occ_idx.view(-1, 1) + add.view(1, -1)

    assert torch.all(edge_neg[0][score_idx].float().std(axis=1) == 0)

    p_neg_by_user = p_neg[score_idx]  # (num_users, num_neg_per_node)
    compare = (p_neg_by_user >= best_p_pos_by_user.view(num_users, 1)).float()
    assert compare.shape == (num_users, num_neg_per_node)
    # compare[i, j], for node i, the j-th negative edge's score > p_best.

    # counts 1 + how many negative edge from src has higher score than p_best.
    # if there's no such negative edge, rank is 1.
    rank_by_user = compare.sum(axis=1) + 1  # (num_users,)
    assert rank_by_user.shape == (num_users,)

    mrr = float(torch.mean(1 / rank_by_user))
    return mrr


# @torch.no_grad()
# def report_rank_based_eval(eval_batch, model, method: str,
#                            num_neg_per_node: int=1000):
#     if num_neg_per_node == -1:
#         # Do not report rank-based metrics, used in debug mode.
#         return 0, 0, 0, 0
#     # Get positive edge indices.
#     edge_index = eval_batch.edge_label_index[:, eval_batch.edge_label == 1]
#     edge_index = edge_index.to('cpu')

#     neg_edge_index = gen_negative_edges(edge_index, num_neg_per_node,
#                                         num_nodes=eval_batch.num_nodes)

#     new_edge_label_index = torch.cat((edge_index, neg_edge_index),
#                                      dim=1).long()
#     new_edge_label = torch.cat((torch.ones(edge_index.shape[1]),
#                                 torch.zeros(neg_edge_index.shape[1])
#                                 ), dim=0).long()

#     # Construct evaluation samples.
#     eval_batch.edge_label_index = new_edge_label_index
#     eval_batch.edge_label = new_edge_label

#     eval_batch.to(torch.device(cfg.device))
#     # move state to gpu
#     for layer in range(len(eval_batch.node_states)):
#         if torch.is_tensor(eval_batch.node_states[layer]):
#             eval_batch.node_states[layer] = eval_batch.node_states[layer].to(
#                 torch.device(cfg.device))
#     pred, true = model(eval_batch)
#     loss, pred_score = compute_loss(pred, true)

#     mrr, recall_at = fast_batch_mrr_and_recall(eval_batch.edge_label_index,
#                                                eval_batch.edge_label,
#                                                pred_score,
#                                                num_neg_per_node,
#                                                eval_batch.num_nodes,
#                                                method)

#     # return mrr, 0, 0, 0
#     #
#     # mrr_old, recall_at_old = compute_src_mrr_and_recall(
#     #     eval_batch.edge_label_index,
#     #     eval_batch.edge_label,
#     #     pred_score,
#     #     recall_k_lst=[1, 3, 10],
#     #     mrr_top_k=1)
#     #
#     # print(f'Old MRR: {mrr_old: 0.6f}, new MRR {mrr: 0.6f}')
#     # print(
#     #     f'Old Recall@1: {recall_at_old[1]: 0.6f}, new Recall@1 {recall_at[1]: 0.6f}')
#     # print(
#     #     f'Old Recall@3: {recall_at_old[3]: 0.6f}, new Recall@3 {recall_at[3]: 0.6f}')
#     # print(
#     #     f'Old Recall@10: {recall_at_old[10]: 0.6f}, new Recall@10 {recall_at[10]: 0.6f}')

#     return mrr, recall_at[1], recall_at[3], recall_at[10]


def get_row_MRR(probs, true_classes):
    existing_mask = true_classes == 1
    # descending in probability for all edge predictions.
    ordered_indices = np.flip(probs.argsort())
    # indicators of positive/negative, in prob desc order.
    ordered_existing_mask = existing_mask[ordered_indices]
    # [1, 2, ... ][ordered_existing_mask]
    # prob rank of positive edges.
    existing_ranks = np.arange(1, true_classes.shape[0] + 1,
                               dtype=np.float)[ordered_existing_mask]
    # average 1/rank of positive edges.
    MRR = (1 / existing_ranks).sum() / existing_ranks.shape[0]
    return MRR


@torch.no_grad()
def report_baseline_MRR(eval_batch: deepsnap.graph.Graph,
                        model: torch.nn.Module) -> float:
    # Get positive edge indices.
    edge_index = eval_batch.edge_label_index[:, eval_batch.edge_label == 1]
    edge_index = edge_index.to('cpu')
    num_nodes = eval_batch.num_nodes
    src_of_pos_edges = torch.unique(edge_index[0]).numpy()

    all_edges_idx = np.arange(num_nodes)
    all_edges_idx = np.array(np.meshgrid(all_edges_idx,
                                         all_edges_idx)).reshape(2, -1)
    all_edges_idx = torch.LongTensor(all_edges_idx)
    # Get all O(N^2) negative edges.
    neg_edge_index = edge_index_difference(
        all_edges_idx, edge_index, num_nodes)
    # Only keep negative edges share src node with some positive edges.
    mask = np.isin(neg_edge_index[0], src_of_pos_edges)
    neg_edge_index = neg_edge_index[:, mask]

    new_edge_label_index = torch.cat((edge_index, neg_edge_index),
                                     dim=1).long()
    new_edge_label = torch.cat((torch.ones(edge_index.shape[1]),
                                torch.zeros(neg_edge_index.shape[1])
                                ), dim=0).long()

    # Construct evaluation samples.
    eval_batch.edge_label_index = new_edge_label_index
    eval_batch.edge_label = new_edge_label

    eval_batch.to(torch.device(cfg.device))
    # move state to gpu
    for layer in range(len(eval_batch.node_states)):
        if torch.is_tensor(eval_batch.node_states[layer]):
            eval_batch.node_states[layer] = eval_batch.node_states[layer].to(
                torch.device(cfg.device))
    pred, true = model(eval_batch)
    loss, pred_score = compute_loss(pred, true)

    probs = pred_score.cpu().numpy().squeeze()
    true = true.cpu().numpy()

    xi = new_edge_label_index[0].cpu().numpy()
    xj = new_edge_label_index[1].cpu().numpy()
    # pred_matrix = coo_matrix((probs, (xi, xj))).toarray()
    # true_matrix = coo_matrix((true, (xi, xj))).toarray()

    row_MRRs = []
    for src in src_of_pos_edges:
        mask = np.argwhere(xi == src)
        pred_row = probs.take(mask).squeeze()
        true_row = true.take(mask).squeeze()
        row_MRRs.append(get_row_MRR(pred_row, true_row))

    # for i, pred_row in enumerate(pred_matrix):
    #     #check if there are any existing edges
    #     # only evaluate senders with existing edge (of course).
    #     if np.isin(1, true_matrix[i]):
    #         row_MRRs.append(get_row_MRR(pred_row, true_matrix[i]))

    avg_MRR = torch.tensor(row_MRRs).mean()
    return float(avg_MRR)


def compute_MRR(eval_batch: deepsnap.graph.Graph,
                model: torch.nn.Module,
                num_neg_per_node: int,
                method: str) -> float:
    """Computes the MRR score on the evaluation batch.

    Args:
        eval_batch (deepsnap.graph.Graph): a graph snapshot.
        model (torch.nn.Module): a GNN model for this graph snapshot
        num_neg_per_node (int): how many negative edges per node required for
            computing the MRR score.
            For example, if num_neg_per_node = 1000, this method firstly
            sample 1,000 negative edges for each source node, and compute the
            average rank of positive edges from each source node among these
            1,000 sampled negative edges.
            Setting num_neg_per_node = -1 to use all possible negative edges.
        method (str): {'min', 'mean', 'max', 'all'}
            All methods firstly compute MRR for each source node, and then
            average MRRs over all source nodes.
            For each source node v,
            let P denote scores of all positive edges from v, the rank()
            operator computes the rank among all negative edges from v.
            'min' computes 1/rank(min(P))
            'mean' computes 1/rank(mean(P))
            'max' computes 1/rank(max(P))
            'all' computes mean[1/rank(x) for x in P]
    """
    if method == 'all':
        # NOTE: this method requires iterating over all nodes, which is slow.
        assert num_neg_per_node == -1
        return report_baseline_MRR(eval_batch, model)
    else:
        assert num_neg_per_node > 0
        # Sample negative edges for each node.
        edge_index = eval_batch.edge_label_index[:, eval_batch.edge_label == 1]
        edge_index = edge_index.to('cpu')

        neg_edge_index = gen_negative_edges(edge_index, num_neg_per_node,
                                            num_nodes=eval_batch.num_nodes)

        new_edge_label_index = torch.cat((edge_index, neg_edge_index),
                                         dim=1).long()
        new_edge_label = torch.cat((torch.ones(edge_index.shape[1]),
                                    torch.zeros(neg_edge_index.shape[1])
                                    ), dim=0).long()

        # Construct evaluation samples.
        eval_batch.edge_label_index = new_edge_label_index
        eval_batch.edge_label = new_edge_label

        eval_batch.to(torch.device(cfg.device))
        # move state to gpu
        for layer in range(len(eval_batch.node_states)):
            if torch.is_tensor(eval_batch.node_states[layer]):
                eval_batch.node_states[layer] = eval_batch.node_states[layer].to(
                    torch.device(cfg.device))
        pred, true = model(eval_batch)
        loss, pred_score = compute_loss(pred, true)

        mrr = fast_batch_mrr(eval_batch.edge_label_index,
                             eval_batch.edge_label,
                             pred_score,
                             num_neg_per_node,
                             eval_batch.num_nodes,
                             method)
        return mrr
