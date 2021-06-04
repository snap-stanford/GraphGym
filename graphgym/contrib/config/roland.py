from yacs.config import CfgNode as CN

from graphgym.register import register_config


def set_cfg_roland(cfg):
    r'''
    This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #
    # Method to update node embedding from old node embedding and new node features.
    # Options: 'moving_average', 'masked_gru', 'gru'
    # moving average: new embedding = r * old + (1-r) * node_feature.
    # gru: new embedding = GRU(node_feature, old_embedding).
    # masked_gru: only apply GRU to active nodes.
    cfg.gnn.embed_update_method = 'moving_average'

    # how many layers to use in the MLP updater.
    # default: 1, use a simple linear layer.
    cfg.gnn.mlp_update_layers = 2
    
    # For meta-learning.
    cfg.meta = CN()
    # Whether to do meta-learning via initialization moving average.
    # Default to False.
    cfg.meta.is_meta = False

    # choose between 'moving_average' and 'online_mean'
    cfg.meta.method = 'moving_average'  # TODO: remove, only use moving_average.
    # For online mean:
    # new_mean = (n-1)/n * old_mean + 1/n * new_value.
    # where *_mean corresponds to W_init.

    # Weight used in moving average for model parameters.
    # After fine-tuning the model in period t and get model M[t],
    # Set W_init = (1-alpha) * W_init + alpha * M[t].
    # For the next period, use W_init as the initialization for fine-tune
    # Set cfg.meta.alpha = 1.0 to recover the original algorithm.
    cfg.meta.alpha = 0.9

    # Use to identify experiments.
    cfg.remark = ''
    # Experimental Features, use this name space to save all controls for
    # experimental features.
    cfg.experimental = CN()

    # How many negative edges for each node to compute rank-based evaluation
    # metrics such as MRR and recall at K.
    # E.g., if multiplier = 1000 and a node has 3 positive edges, then we
    # compute the MRR using 1000 randomly generated negative edges
    # + 3 existing positive edges.
    cfg.experimental.rank_eval_multiplier = 1000

    # Only use the first n snapshots (time periods) to train the model.
    # Empirically, the model learns rich dynamics from only a few periods.
    # Set to -1 if using all snapshots.
    cfg.experimental.restrict_training_set = -1

    # Whether to visualize edge attention of GNN layer after training.
    cfg.experimental.visualize_gnn_layer = False

    cfg.train.tbptt_freq = 5

    cfg.train.internal_validation_tolerance = 5

    # Computing MRR is slow in the baseline setting.
    # Only start to compute MRR in the test set range after certain time.
    cfg.train.start_compute_mrr = 0
    
    # How to handle node features in AS dataset.
    # available: ['one', 'one_hot_id', 'one_hot_degree_global', 'one_hot_degree_local']
    cfg.dataset.AS_node_feature = 'one'

    # ----------------------------------------------------------------------- #
    # Additional dataset option for the BSI dataset.
    # ----------------------------------------------------------------------- #
    # Method used to sample negative edges for edge_label_index.
    # 'uniform': all non-existing edges have same probability of being sampled
    #            as negative edges.
    # 'src':  non-existing edges from high-degree nodes are more likely to be
    #         sampled as negative edges.
    # 'dest': non-existing edges pointed to high-degree nodes are more likely
    #         to be sampled as negative edges.
    cfg.dataset.negative_sample_weight = 'uniform'

    # whether to load heterogeneous graphs.
    cfg.dataset.is_hetero = False

    # where to put type information. 'append' or 'graph_attribute'.
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

    cfg.gnn.skip_connection = 'none'  # {'none', 'identity', 'affine'}
    # ----------------------------------------------------------------------- #
    # Customized options
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

    cfg.metric = CN()
    # how to compute MRR.
    # available: f = 'min', 'max', 'mean'.
    # Step 1: get the p* = f(scores of positive edges)
    # Step 2: compute the rank r of p* among all negative edges.
    # Step 3: RR = 1 / rank.
    # Step 4: average over all users.
    # expected MRR(min) <= MRR(mean) <= MRR(max).
    cfg.metric.mrr_method = 'max'

    # Specs for the link prediction task using BSI dataset.
    # All units are days.
    cfg.link_pred_spec = CN()

    # The period of `today`'s increase: how often the system is making forecast.
    # E.g., when = 1,
    # the system forecasts transactions in upcoming 7 days for everyday.
    # One training epoch loops over
    # {Jan-1-2020, Jan-2-2020, Jan-3-2020..., Dec-31-2020}
    # When = 7, the system makes prediction every week.
    # E.g., the system forecasts transactions in upcoming 7 days
    # on every Monday.
    cfg.link_pred_spec.forecast_frequency = 1

    # How many days into the future the model is trained to predict.
    # The model forecasts transactions in (today, today + forecast_horizon].
    # NOTE: forecast_horizon should >= forecast_frequency to cover all days.
    cfg.link_pred_spec.forecast_horizon = 7


register_config('roland', set_cfg_roland)
