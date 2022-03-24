import torch
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

import graphgym.register as register

# Used for the OGB Encoders
full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()


class IntegerFeatureEncoder(torch.nn.Module):
    """
    Provides an encoder for integer node features.

    Args:
        emb_dim (int): Output embedding dimension
        num_classes (int): the number of classes for the
        embedding mapping to learn from
    """
    def __init__(self, emb_dim, num_classes=None):
        super(IntegerFeatureEncoder, self).__init__()

        self.encoder = torch.nn.Embedding(num_classes, emb_dim)
        torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        # Encode just the first dimension if more exist
        batch.node_feature = self.encoder(batch.node_feature[:, 0])

        return batch


class SingleAtomEncoder(torch.nn.Module):
    """
    Only encode the first dimension of atom integer features.
    This feature encodes just the atom type
    Args:
        emb_dim (int): Output embedding dimension
        num_classes: None
    """
    def __init__(self, emb_dim, num_classes=None):
        super(SingleAtomEncoder, self).__init__()

        num_atom_types = full_atom_feature_dims[0]
        self.atom_type_embedding = torch.nn.Embedding(num_atom_types, emb_dim)
        torch.nn.init.xavier_uniform_(self.atom_type_embedding.weight.data)

    def forward(self, batch):
        batch.node_feature = self.atom_type_embedding(batch.node_feature[:, 0])

        return batch


class AtomEncoder(torch.nn.Module):
    """
    The atom Encoder used in OGB molecule dataset.

    Args:
        emb_dim (int): Output embedding dimension
        num_classes: None
    """
    def __init__(self, emb_dim, num_classes=None):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, batch):
        encoded_features = 0
        for i in range(batch.node_feature.shape[1]):
            encoded_features += self.atom_embedding_list[i](
                batch.node_feature[:, i])
        batch.node_feature = encoded_features
        return batch


class BondEncoder(torch.nn.Module):
    """
    The bond Encoder used in OGB molecule dataset.

    Args:
        emb_dim (int): Output edge embedding dimension
    """
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, batch):
        bond_embedding = 0
        for i in range(batch.edge_feature.shape[1]):
            bond_embedding += self.bond_embedding_list[i](
                batch.edge_feature[:, i])

        batch.edge_feature = bond_embedding
        return batch


node_encoder_dict = {
    'Integer': IntegerFeatureEncoder,
    'SingleAtom': SingleAtomEncoder,
    'Atom': AtomEncoder
}

node_encoder_dict = {**register.node_encoder_dict, **node_encoder_dict}

edge_encoder_dict = {'Bond': BondEncoder}

edge_encoder_dict = {**register.edge_encoder_dict, **edge_encoder_dict}
