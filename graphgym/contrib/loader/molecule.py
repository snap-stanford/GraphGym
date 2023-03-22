# MIT License
# Copyright (c) 2020 Jiaxuan You, Wengong Jin, Octavian Ganea,
# Regina Barzilay, Tommi Jaakkola
# Copyright (c) 2020 Wengong Jin, Kyle Swanson, Kevin Yang,
# Regina Barzilay, Tommi Jaakkola
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import csv
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Set, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from deepsnap.graph import Graph
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import qed
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

dirname = os.path.dirname(os.path.abspath(__file__))

# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num':
    list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(
    range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values;
# + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14


def get_atom_fdim() -> int:
    """Gets the dimensionality of atom features."""
    return ATOM_FDIM


def get_bond_fdim(atom_messages: bool = True) -> int:
    """
    Gets the dimensionality of bond features.

    :param atom_messages whether atom messages are being used. If atom
        messages, only contains bond features.
        Otherwise contains both atom and bond features.
    :return: The dimensionality of bond features.
    """
    return BOND_FDIM + (not atom_messages) * get_atom_fdim()


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of
        length len(choices) + 1.
        If value is not in the list of choices,
        then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(
        atom: Chem.rdchem.Atom,
        functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups
        the atom belongs to.
    :return: A list containing the atom features.
    """
    # todo: consider add isinring feature
    features = onek_encoding_unk(
        atom.GetAtomicNum() - 1,
        ATOM_FEATURES['atomic_num']) + onek_encoding_unk(
            atom.GetTotalDegree(),
            ATOM_FEATURES['degree']) + onek_encoding_unk(
                atom.GetFormalCharge(),
                ATOM_FEATURES['formal_charge']) + onek_encoding_unk(
                    int(atom.GetChiralTag()),
                    ATOM_FEATURES['chiral_tag']) + onek_encoding_unk(
                        int(atom.GetTotalNumHs()),
                        ATOM_FEATURES['num_Hs']) + onek_encoding_unk(
                            int(atom.GetHybridization()),
                            ATOM_FEATURES['hybridization']) + [
                                1 if atom.GetIsAromatic() else 0
                            ] + [atom.GetMass() * 0.01]  # scale to same range
    if functional_groups is not None:
        features += functional_groups
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


def get_fname(path):
    return os.path.basename(path).split('.')[0]


def get_graph_stats(g):
    print(g.number_of_nodes(), g.number_of_edges(), nx.average_clustering(g),
          nx.diameter(g))


def mol2nx(mol):
    G = nx.Graph()
    for atom in mol.GetAtoms():
        index = atom.GetIdx()
        feature = torch.tensor(atom_features(atom))
        G.add_node(index, node_feature=feature)
    for bond in mol.GetBonds():
        index_begin = bond.GetBeginAtomIdx()
        index_end = bond.GetEndAtomIdx()
        feature = torch.tensor(bond_features(bond))
        G.add_edge(index_begin, index_end, edge_feature=feature)
    return G


def mol2data(mol):
    n = mol.GetNumAtoms()
    nd = get_atom_fdim()
    e = mol.GetNumBonds()
    ed = get_bond_fdim()
    node_feature = torch.zeros((n, nd))
    edge_index = torch.zeros((2, e), dtype=torch.long)
    edge_feature = torch.zeros((e, ed))
    for atom in mol.GetAtoms():
        index = atom.GetIdx()
        node_feature[index, :] = torch.tensor(atom_features(atom))
    for i, bond in enumerate(mol.GetBonds()):
        index_begin = bond.GetBeginAtomIdx()
        index_end = bond.GetEndAtomIdx()
        edge_index[:, i] = torch.tensor([index_begin, index_end])
        edge_feature[i, :] = torch.tensor(bond_features(bond))
    edge_index = torch.cat((edge_index, torch.flip(edge_index, [0])), dim=1)
    edge_feature = torch.cat((edge_feature, edge_feature), dim=0)

    data = Graph(node_feature=node_feature,
                 edge_index=edge_index,
                 edge_feature=edge_feature)

    return data


def smiles2nx(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol2nx(mol)


def smiles2data(smiles, return_mol=False):
    mol = Chem.MolFromSmiles(smiles)
    if return_mol:
        return mol2data(mol), mol
    else:
        return mol2data(mol)


def mols2graphs(all_mols):
    graphs = []
    for smiles, val in all_mols.items():
        G = smiles2nx(smiles)
        if G.number_of_edges() == 0:
            continue
        for key in val.keys():
            G.graph[key] = val[key]
        graphs.append(G)
    return graphs


def mols2data(all_mols, return_scaffold_split=False):
    all_data = []
    all_data_mol = []
    for smiles, val in all_mols.items():
        data, mol = smiles2data(smiles, return_mol=True)
        if data.num_edges == 0:
            continue
        if isinstance(val, dict):
            for key in val.keys():
                data['graph_{}'.format(key)] = torch.tensor(val[key])
        all_data.append(data)
        all_data_mol.append(mol)
    if return_scaffold_split:
        splits = mol_scaffold_split(all_data_mol)
        return all_data, splits
    else:
        return all_data


# Load data
def load_mol_analysis(path_list):
    all_smiles = []
    all_smiles_unique = set()
    for path in path_list:
        smiles_unique = set()
        with open(path) as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
            smiles_column = columns[0]
            for row in tqdm(reader):
                smiles = row[smiles_column]
                mol = Chem.MolFromSmiles(smiles)
                smiles = Chem.MolToSmiles(mol)  # canonical smiles
                smiles_unique.add(smiles)
        all_smiles += list(smiles_unique)
        all_smiles_unique = all_smiles_unique.union(smiles_unique)

    # Plot
    smile_count = dict(Counter(all_smiles))

    counts = []
    for smile, count in smile_count.items():
        if count > 1:
            counts.append(count)
            # print(smile, count)
    print(len(all_smiles_unique), len(counts))

    count, bins = np.histogram(np.array(counts), bins=np.arange(2, 16))
    plt.figure()
    plt.plot(bins[1:], count)
    plt.show()
    plt.figure()
    plt.plot(bins[1:], np.log10(count))
    plt.show()


class Molecule(object):
    def __init__(self, smiles):
        self.mol = Chem.MolFromSmiles(smiles)
        self.smiles = Chem.MolToSmiles(self.mol)  # canonical smiles

    def build_graph(self):
        self.graph = smiles2nx(self.smiles)


def min_max_scaler(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)


# Load data
def load_mol_datasets(name_list, use_cache=False):
    # todo: using pandas may be more efficient

    path_list = ['{}/data/{}.csv'.format(dirname, name) for name in name_list]

    cache_mols_name = '{}/cache/mols_{}.pt'.format(dirname,
                                                   '_'.join(name_list))
    cache_splits_name = '{}/cache/splits_{}.pt'.format(dirname,
                                                       '_'.join(name_list))
    cache_motifs_name = '{}/cache/motifs_{}.pt'.format(dirname,
                                                       '_'.join(name_list))
    cache_targets_name = '{}/cache/targets_{}.pt'.format(
        dirname, '_'.join(name_list))

    if not use_cache or not os.path.isfile(
            cache_mols_name
    ) or not os.path.isfile(cache_splits_name) or not os.path.isfile(
            cache_motifs_name) or not os.path.isfile(cache_targets_name):

        # get all target names
        all_targets = ['logp', 'qed']
        for path in path_list:
            with open(path) as f:
                fname = get_fname(path)
                reader = csv.DictReader(f)
                columns = reader.fieldnames
                target_columns = columns[1:]
                target_columns = [
                    '{}_{}'.format(fname, target_column)
                    for target_column in target_columns
                ]
                all_targets += target_columns

        target_id_bias = 2
        all_mols = {}

        motifs = {}
        logp_all = []
        qed_all = []
        for path in path_list:
            with open(path) as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames
                smiles_column = columns[0]
                target_columns = columns[1:]

                smiles_unique = set()
                for row in tqdm(reader):
                    smiles = row[smiles_column]
                    mol = Chem.MolFromSmiles(smiles)
                    smiles = Chem.MolToSmiles(mol)  # canonical smiles
                    if smiles in smiles_unique or mol is None \
                            or mol.GetNumAtoms() <= 1:
                        continue
                    smiles_unique.add(smiles)

                    motif_mols, motif_smiles = find_fragments(mol)
                    # update overall motifs dict
                    for smiles_temp in motif_smiles:
                        if smiles_temp not in motifs:
                            motifs[smiles_temp] = len(motifs)
                    motifs_id = [
                        motifs[smiles_temp] for smiles_temp in motif_smiles
                    ]

                    targets_id = []
                    targets_value = []
                    # add synthetic molecule feature
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        value_logp = MolLogP(mol)
                        targets_id.append(0)
                        targets_value.append(value_logp)
                        logp_all.append(value_logp)
                        value_qed = qed(mol)
                        targets_id.append(1)
                        targets_value.append(value_qed)
                        qed_all.append(value_qed)
                    except Exception:
                        print('--cannot compute logp or qed')

                    for id, column in enumerate(target_columns):
                        if row[column] != '':
                            id_all = id + target_id_bias
                            value = float(row[column])
                            targets_id.append(id_all)
                            targets_value.append(value)
                    if mol is None:
                        print(smiles)
                    if all_mols.get(smiles) is None:
                        all_mols[smiles] = {
                            'targets_id': targets_id,
                            'targets_value': targets_value,
                            'motifs_id': motifs_id
                        }
                    else:
                        all_mols[smiles]['targets_id'] += targets_id
                        all_mols[smiles]['targets_value'] += targets_value
                        all_mols[smiles]['motifs_id'] += motifs_id
            target_id_bias += len(target_columns)
        # normalize logp and qed
        logp_all_min, logp_all_max = min(logp_all), max(logp_all)
        qed_all_min, qed_all_max = min(qed_all), max(qed_all)
        for smiles in all_mols.keys():
            for i, targets_id in enumerate(all_mols[smiles]['targets_id']):

                if targets_id == 0:
                    all_mols[smiles]['targets_value'][i] = min_max_scaler(
                        all_mols[smiles]['targets_value'][i], logp_all_min,
                        logp_all_max)
                elif targets_id == 1:
                    all_mols[smiles]['targets_value'][i] = min_max_scaler(
                        all_mols[smiles]['targets_value'][i], qed_all_min,
                        qed_all_max)

        all_mols, splits = mols2data(all_mols, return_scaffold_split=True)
        all_motifs = mols2data(motifs)

        torch.save(all_mols, cache_mols_name)
        torch.save(splits, cache_splits_name)
        torch.save(all_motifs, cache_motifs_name)
        torch.save(all_targets, cache_targets_name)
    else:
        all_mols = torch.load(cache_mols_name)
        splits = torch.load(cache_splits_name)
        all_motifs = torch.load(cache_motifs_name)
        all_targets = torch.load(cache_targets_name)

    return all_mols, splits, all_motifs, all_targets


def copy_atom(atom, atommap=True):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    if atommap:
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom


def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        Chem.Kekulize(mol)
    return mol


def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


def sanitize(mol, kekulize=True):
    try:
        smiles = get_smiles(mol) if kekulize else Chem.MolToSmiles(mol)
        mol = get_mol(smiles) if kekulize else Chem.MolFromSmiles(smiles)
    except Exception:
        mol = None
    return mol


def get_clique_mol(mol, atoms):
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol)
    return new_mol


def find_fragments(mol):
    new_mol = Chem.RWMol(mol)
    for atom in new_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())

    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        if a1.IsInRing() and a2.IsInRing():
            new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())

        elif a1.IsInRing() and a2.GetDegree() > 1:
            new_idx = new_mol.AddAtom(copy_atom(a1))
            new_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(a1.GetIdx())
            new_mol.AddBond(new_idx, a2.GetIdx(), bond.GetBondType())
            new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())

        elif a2.IsInRing() and a1.GetDegree() > 1:
            new_idx = new_mol.AddAtom(copy_atom(a2))
            new_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(a2.GetIdx())
            new_mol.AddBond(new_idx, a1.GetIdx(), bond.GetBondType())
            new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())

    new_mol = new_mol.GetMol()
    new_smiles = Chem.MolToSmiles(new_mol)

    motif_mols = []
    motif_smiles = []
    for fragment in new_smiles.split('.'):
        fmol = Chem.MolFromSmiles(fragment)
        if fmol is None:
            continue
        indices = set([atom.GetAtomMapNum() for atom in fmol.GetAtoms()])
        fmol = get_clique_mol(mol, indices)
        if fmol is None:
            continue
        fsmiles = Chem.MolToSmiles(fmol)
        fmol = Chem.MolFromSmiles(fsmiles)
        if fmol is None or fmol.GetNumBonds() < 1 or fmol.GetNumAtoms() <= 1:
            continue
        if fsmiles in motif_smiles:
            continue
        motif_mols.append(fmol)
        motif_smiles.append(fsmiles)

    # at least return 1 motif
    if len(motif_mols) == 0:
        motif_mols = [mol]
        motif_smiles = [Chem.MolToSmiles(mol)]

    return motif_mols, motif_smiles


def generate_scaffold(mol: Union[str, Chem.Mol],
                      include_chirality: bool = False) -> str:
    """
    Compute the Bemis-Murcko scaffold for a SMILES string.

    :param mol: A smiles string or an RDKit molecule.
    :param include_chirality: Whether to include chirality.
    :return:
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=include_chirality)

    return scaffold


def scaffold_to_smiles(
        mols: Union[List[str], List[Chem.Mol]],
        use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes scaffold for each smiles string and returns a mapping from
    scaffolds to sets of smiles.

    :param mols: A list of smiles strings or RDKit molecules.
    :param use_indices: Whether to map to the smiles' index in all_smiles
    rather than mapping to the smiles string itself. This is necessary if there
    are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all smiles
    (or smiles indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds


def mol_scaffold_split(data, sizes=[0.8, 0.1, 0.1], balanced=True, repeat=10):
    '''

    :param data: all_mols
    :param sizes: A length-3 tuple with the proportions of data in the
    train, validation, and test sets.
    :param balanced: Try to balance sizes of scaffolds in each set, rather than
    just putting smallest in test set.
    :return:
    '''
    assert sum(sizes) == 1

    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles(data, use_indices=True)
    train_all = []
    val_all = []
    test_all = []

    for i in range(repeat):
        random.seed(i)
        # Split
        train_size, val_size, test_size = sizes[0] * len(data), sizes[1] * len(
            data), sizes[2] * len(data)
        train, val, test = [], [], []
        train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0
        # Put stuff that's bigger than half the val/test size into train,
        # rest just order randomly
        if balanced:
            index_sets = list(scaffold_to_indices.values())
            big_index_sets = []
            small_index_sets = []
            for index_set in index_sets:
                if len(index_set) > val_size / 2 or len(
                        index_set) > test_size / 2:
                    big_index_sets.append(index_set)
                else:
                    small_index_sets.append(index_set)
            random.shuffle(big_index_sets)
            random.shuffle(small_index_sets)
            index_sets = big_index_sets + small_index_sets
        else:  # Sort from largest to smallest scaffold sets
            index_sets = sorted(list(scaffold_to_indices.values()),
                                key=lambda index_set: len(index_set),
                                reverse=True)

        for index_set in index_sets:
            if len(train) + len(index_set) <= train_size:
                train += index_set
                train_scaffold_count += 1
            elif len(val) + len(index_set) <= val_size:
                val += index_set
                val_scaffold_count += 1
            else:
                test += index_set
                test_scaffold_count += 1
        train_all.append(train)
        val_all.append(val)
        test_all.append(test)

    return {'train': train_all, 'valid': val_all, 'test': test_all}
