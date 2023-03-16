import json
import h5py
import time
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import dgl
import dgl.function as fn
from dgl.nn import SAGEConv
from sklearn.metrics import roc_auc_score

# two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')
    
    def forward(self, g, in_feats):
        h = self.conv1(g, in_feats)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
    
class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]
        
class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']
        
def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

def main(per_protein_file, aso_file, ppi_file, no_nf):
    seq = []
    emb = []
    emb_mean = []
    with h5py.File(per_protein_file, "r") as file:
        print(f"number of entries: {len(file.items())}")
        for sequence_id, embedding in file.items():
            seq.append(sequence_id)
            emb.append(embedding)
            emb_mean.append(np.array(embedding).mean())

    f = open(aso_file)
    gene_to_id = json.load(f)

    nf_dict= {}
    for i, gene in enumerate(gene_to_id):
        gid = gene_to_id[gene]
        for j, s in enumerate(seq):
            if gid != '-':
                if s == gid:
                    nf_dict[gene] = emb_mean[j]

    ppi = pd.read_csv(ppi_file).head(5000)
    all_nodes = list(set(list(ppi['gene_a'].unique()) + list(ppi['gene_b'].unique())))
    g = dgl.DGLGraph()
    g.add_nodes(len(all_nodes))

    id_dict = {}
    for i, gene in enumerate(all_nodes):
        id_dict[gene] = i

    genea_idx = []
    for i, gene in enumerate(ppi['gene_a']):
        for j, idx_match in enumerate(id_dict):
            if gene == idx_match:
                genea_idx.append(id_dict[idx_match])
    geneb_idx = []
    for i, gene in enumerate(ppi['gene_b']):
        for j, idx_match in enumerate(id_dict):
            if gene == idx_match:
                geneb_idx.append(id_dict[idx_match])

    g.add_edges(torch.tensor(genea_idx).long(), torch.tensor(geneb_idx).long())

    cell_cat = torch.tensor(pd.get_dummies(ppi['cell_category']).values)
    cell_sex = torch.tensor(pd.get_dummies(ppi['cell_sex']).values)
    cell_spe = torch.tensor(pd.get_dummies(ppi['cell_species']).values)

    ef = torch.cat([cell_cat, cell_sex, cell_spe], dim=-1).float()

    feat = []
    id_feat = []
    ids = []
    for i, gene_id in enumerate(id_dict.keys()):
        for j, gene_nf in enumerate(nf_dict.keys()): 
            if gene_nf == gene_id:
                feat.append([nf_dict[gene_nf]])
                id_feat.append([id_dict[gene_id], nf_dict[gene_nf]])
                ids.append(id_dict[gene_id])
                
    node_remove = []
    for i in range(len(all_nodes)):
        if i not in ids:
            node_remove.append(i)

    if no_nf:
        g.edata['feat'] = ef
        g.ndata['_feat'] = torch.zeros(g.num_nodes(), ef.size(1))
        node_dim = g.ndata['_feat'].size(1)
        edge_dim = g.edata['feat'].size(1)
        latent_dim = 5
        node_encoder = nn.Linear(node_dim, latent_dim)
        edge_encoder = nn.Linear(edge_dim, latent_dim)
        g.ndata['_h'] = node_encoder(g.ndata['_feat'])
        g.edata['_h'] = edge_encoder(g.edata['feat'])
        g.pull(g.nodes(),
            message_func=fn.copy_e('feat', 'm'),
            reduce_func=fn.sum('m', 'feat'))
    else:
        g = dgl.remove_nodes(g, node_remove)
        g.ndata['feat'] = torch.tensor(feat).float()


    # Split edge set for training and testing
    u, v = g.edges()

    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    train_size = g.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(g.number_of_nodes(), g.number_of_nodes()))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    train_g = dgl.remove_edges(g, eids[:test_size])

    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

    model = GraphSAGE(train_g.ndata['feat'].shape[1], 16)
    # You can replace DotPredictor with MLPPredictor.
    #pred = MLPPredictor(16)
    pred = DotPredictor()

    # in this case, loss will in training loop
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)

    # ----------- training -------------------------------- #
    all_logits = []
    for e in range(50):
        # forward
        h = model(train_g, train_g.ndata['feat'])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print('In epoch {}, loss: {}'.format(e, loss))

    # ----------- check results ------------------------ #
    from sklearn.metrics import roc_auc_score
    with torch.no_grad():
        pos_score = pred(test_pos_g, h)
        neg_score = pred(test_neg_g, h)
        print('AUC', compute_auc(pos_score, neg_score))

if __name__ == '__main__':
    main(per_protein_file="rna/data/per-protein.h5",
         aso_file='rna/aso_data/homo_gene2acc_json.json',
         ppi_file='rna/context-PPI_final.csv',
         no_nf=True)