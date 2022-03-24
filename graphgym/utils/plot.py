import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

from graphgym.utils.io import makedirs

# from sklearn.manifold import TSNE


sns.set_context('poster')


def view_emb(emb, dir):
    '''
    Visualize a embedding matrix.

    Args:
        emb (torch.tensor): Embedding matrix with shape (N, D). D is the
        feature dimension.
        dir (str): Output directory for the embedding figure.

    '''
    if emb.shape[1] > 2:
        pca = PCA(n_components=2)
        emb = pca.fit_transform(emb)
    plt.figure(figsize=(10, 10))
    plt.scatter(emb[:, 0], emb[:, 1])
    plt.savefig('{}/emb_pca.png'.format(dir), dpi=100)


def view_emb_kg(emb1, emb2, dir, epoch=0):
    pca = PCA(n_components=2)
    emb = np.concatenate((emb1, emb2), axis=0)
    print(emb.shape)
    split = emb1.shape[0]
    emb = pca.fit_transform(emb)
    plt.figure(figsize=(10, 10))
    plt.scatter(emb[:split, 0], emb[:split, 1], c='green', s=100)
    plt.scatter(emb[split:, 0], emb[split:, 1], c='blue', marker='x', s=800)
    ax = plt.gca()
    annotate = {-3: 'LogP', -2: 'QED', -1: 'Label'}
    for i, txt in annotate.items():
        ax.annotate(txt, (emb[i, 0], emb[i, 1]))
    makedirs('{}/emb'.format(dir))
    plt.savefig('{}/emb/pca_{}.png'.format(dir, epoch), dpi=100)
