from yacs.config import CfgNode as CN

from graphgym.register import register_config


def set_cfg_example(cfg):
    r'''
    This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    '''

    # ----------------------------------------------------------------------- #
    # MetaLink KG options
    # ----------------------------------------------------------------------- #

    cfg.kg = CN()

    cfg.kg.kg_mode = True

    cfg.kg.dim_emb = 64

    # type of prediction head: direct, mp, bilinear
    cfg.kg.head = 'direct'

    # how to decode pred
    cfg.kg.decode = 'dot'

    cfg.kg.layer_type = 'kgheteconv'

    # normalize embedding
    cfg.kg.norm_emb = False

    # if do fine_tune after training
    cfg.kg.fine_tune = False

    # kg message passing layers
    cfg.kg.layers_mp = 0

    # kg aggregation
    cfg.kg.agg = 'mean'

    # kg message direction
    cfg.kg.msg_direction = 'single'

    # kg gate function
    cfg.kg.gate_self = True
    cfg.kg.gate_msg = True

    # kg self transform
    cfg.kg.self_trans = True

    # Add self msg passing
    cfg.kg.self_msg = 'none'

    cfg.kg.has_act = True  # not sure

    cfg.kg.has_bn = False  # picked

    cfg.kg.hete = True  # picked

    # last, every
    cfg.kg.pred = 'every'  # picked

    # no, every, last
    cfg.kg.has_l2norm = 'no'  # picked, every if needed

    # positioning l2 norm: pre, post
    cfg.kg.pos_l2norm = 'post'

    cfg.kg.gate_bias = False

    # raw, loss, both
    cfg.kg.edge_feature = 'raw'

    # pertask, standard
    cfg.kg.split = 'pertask'

    # new, standard
    cfg.kg.experiment = 'new'

    # standard, relation
    cfg.kg.setting = 'standard'

    # Whether do meta inductive learning
    cfg.kg.meta = False

    # cfg.kg.meta_num = None

    cfg.kg.meta_ratio = 0.2

    # whether add aux target (logp, qed)
    cfg.kg.aux = True

    # keep what percentage of edges
    cfg.kg.setting_ratio = 0.5

    # number of repeats for evaluation
    cfg.kg.repeat = 1


register_config('metalink', set_cfg_example)
