from yacs.config import CfgNode as CN

from graphgym.register import register_config


def set_cfg_roland(cfg):
    """
    This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    """

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    # Use to identify experiments, tensorboard will be saved to this path.
    # Options: any string.
    cfg.remark = ''

    # ----------------------------------------------------------------------- #
    # Additional GNN options.
    # ----------------------------------------------------------------------- #
    # Method to update node embedding from old node embedding and new node features.
    # Options: {'moving_average', 'mlp', 'gru'}
    cfg.gnn.embed_update_method = 'moving_average'

    # How many layers to use in the MLP updater.
    # Options: integers >= 1.
    # NOTE: there is a known issue when set to 1, use >= 2 for now.
    # Only effective when cfg.gnn.embed_update_method == 'mlp'.
    cfg.gnn.mlp_update_layers = 2

    # What kind of skip-connection to use.
    # Options: {'none', 'identity', 'affine'}.
    cfg.gnn.skip_connection = 'none'

    # ----------------------------------------------------------------------- #
    # Meta-Learning options.
    # ----------------------------------------------------------------------- #
    # For meta-learning.
    cfg.meta = CN()
    # Whether to do meta-learning via initialization moving average.
    # Options: {True, False}
    cfg.meta.is_meta = False

    # Weight used in moving average for model parameters.
    # After fine-tuning the model in period t and get model M[t],
    # Set W_init = (1-alpha) * W_init + alpha * M[t].
    # For the next period, use W_init as the initialization for fine-tune
    # Set cfg.meta.alpha = 1.0 to recover the original algorithm.
    # Options: float between 0.0 and 1.0.
    cfg.meta.alpha = 0.9

    # ----------------------------------------------------------------------- #
    # Additional GNN options.
    # ----------------------------------------------------------------------- #
    # How many snapshots for the truncated back-propagation.
    # Set to a very large integer to use full-back-prop-through-time
    # Options: integers >= 1.
    cfg.train.tbptt_freq = 10

    # Early stopping tolerance in live-update.
    # Options: integers >= 1.
    cfg.train.internal_validation_tolerance = 5

    # Computing MRR is slow in the baseline setting.
    # Only start to compute MRR in the test set range after certain time.
    # Options: integers >= 0.
    cfg.train.start_compute_mrr = 0

    # ----------------------------------------------------------------------- #
    # Additional dataset options.
    # ----------------------------------------------------------------------- #

    # How to handle node features in AS-733 dataset.
    # Options: ['one', 'one_hot_id', 'one_hot_degree_global']
    cfg.dataset.AS_node_feature = 'one'

    # Method used to sample negative edges for edge_label_index.
    # Options:
    # 'uniform': all non-existing edges have same probability of being sampled
    #            as negative edges.
    # 'src':  non-existing edges from high-degree nodes are more likely to be
    #         sampled as negative edges.
    # 'dest': non-existing edges pointed to high-degree nodes are more likely
    #         to be sampled as negative edges.
    cfg.dataset.negative_sample_weight = 'uniform'

    # Whether to load dataset as heterogeneous graphs.
    # Options: {True, False}.
    cfg.dataset.is_hetero = False

    # Where to put type information.
    # Options: {'append', 'graph_attribute'}.
    # Only effective if cfg.dataset.is_hetero == True.
    cfg.dataset.type_info_loc = 'append'

    # whether to look for and load cached graph. By default (load_cache=False)
    # the loader loads the raw tsv file from disk and
    cfg.dataset.load_cache = False

    cfg.dataset.premade_datasets = 'fresh'

    cfg.dataset.include_node_features = False

    # 'chronological_temporal' or 'default'.
    # 'chronological_temporal': only for temporal graphs, for example,
    # the first 80% snapshots are for training, then subsequent 10% snapshots
    # are for validation and the last 10% snapshots are for testing.
    cfg.dataset.split_method = 'default'

    # ----------------------------------------------------------------------- #
    # Customized options: `transaction` for ROLAND dynamic graphs.
    # ----------------------------------------------------------------------- #

    # example argument group
    cfg.transaction = CN()

    # whether use snapshot
    cfg.transaction.snapshot = False

    # snapshot split method 1: number of snapshots
    # split dataset into fixed number of snapshots.
    cfg.transaction.snapshot_num = 100

    # snapshot split method 2: snapshot frequency
    # e.g., one snapshot contains transactions within 1 day.
    cfg.transaction.snapshot_freq = 'D'

    cfg.transaction.check_snapshot = False

    # how to use transaction history
    # full or rolling
    cfg.transaction.history = 'full'

    # type of loss: supervised / meta
    cfg.transaction.loss = 'meta'

    # feature dim for int edge features
    cfg.transaction.feature_int_dim = 32
    cfg.transaction.feature_edge_int_num = [50, 8, 252, 252, 3, 3]
    cfg.transaction.feature_node_int_num = [0]

    # feature dim for amount (float) edge feature
    cfg.transaction.feature_amount_dim = 64

    # feature dim for time (float) edge feature
    cfg.transaction.feature_time_dim = 64

    #
    cfg.transaction.node_feature = 'raw'

    # how many days look into the future
    cfg.transaction.horizon = 1

    # prediction mode for the task; 'before' or 'after'
    cfg.transaction.pred_mode = 'before'

    # number of periods to be captured.
    # set to a list of integers if wish to use pre-defined periodicity.
    # e.g., [1,7,28,31,...] etc.
    cfg.transaction.time_enc_periods = [1]

    # if 'enc_before_diff': attention weight = diff(enc(t1), enc(t2))
    # if 'diff_before_enc': attention weight = enc(t1 - t2)
    cfg.transaction.time_enc_mode = 'enc_before_diff'

    # how to compute the keep ratio while updating the recurrent GNN.
    # the update ratio (for each node) is a function of its degree in [0, t)
    # and its degree in snapshot t.
    cfg.transaction.keep_ratio = 'linear'

    # ----------------------------------------------------------------------- #
    # Customized options: metrics.
    # ----------------------------------------------------------------------- #

    cfg.metric = CN()
    # How many negative edges for each node to compute rank-based evaluation
    # metrics such as MRR and recall at K.
    # E.g., if multiplier = 1000 and a node has 3 positive edges, then we
    # compute the MRR using 1000 randomly generated negative edges
    # + 3 existing positive edges.
    # Use 100 ~ 1000 for fast and reliable results.
    cfg.metric.mrr_num_negative_edges = 1000

    # how to compute MRR.
    # available: f = 'min', 'max', 'mean'.
    # Step 1: get the p* = f(scores of positive edges)
    # Step 2: compute the rank r of p* among all negative edges.
    # Step 3: RR = 1 / rank.
    # Step 4: average over all users.
    # expected MRR(min) <= MRR(mean) <= MRR(max).
    cfg.metric.mrr_method = 'max'


register_config('roland', set_cfg_roland)
