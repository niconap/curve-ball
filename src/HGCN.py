import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
import os
from torch_geometric.utils import dense_to_sparse

from models.hyper_decoder import FermiDiracDecoder, SoftmaxDecoder
import manifolds
import models.hyper_encoder as encoders
import models.hyper_decoder as decoders
from torch_geometric.nn import GCNConv
from metrics.train_metrics import HGCNTrainLoss
import utils
import pdb

class HGCN(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features, domain_features):
        super(HGCN, self).__init__()
        
        self.cfg = cfg
        self.c = cfg.model.c
        self.dataset_infos = dataset_infos
        input_dim = dataset_infos.input_dims
        output_dim = dataset_infos.output_dims
        self.manifold_name = cfg.model.manifold
        self.manifold = getattr(manifolds, self.manifold_name)()
        
        self.encoder = getattr(encoders, cfg.model.encoder)(cfg.model.c, cfg)
        self.fd_decoder = FermiDiracDecoder(cfg.model.r, cfg.model.t)
        # self.fd_decoder = SoftmaxDecoder(cfg.model.r, cfg.model.t)
        self.nc_decoder = getattr(decoders, cfg.model.decoder)(cfg.model.c, cfg)
        self.lp_decoder = getattr(decoders, cfg.model.lpdecoder)(cfg.model.c, cfg)
        
        self.train_loss = HGCNTrainLoss(cfg.model.lambda_node, cfg.model.lambda_edge)
        self.val_loss = HGCNTrainLoss(cfg.model.lambda_node, cfg.model.lambda_edge)
        self.test_loss = HGCNTrainLoss(cfg.model.lambda_node, cfg.model.lambda_edge)
        self.train_metrics = train_metrics
        # pdb.set_trace()
        
    def edge_decode(self, h, idx, num_nodes, batch_index):
        """
        Decode edges with batch support.
        h: Node embeddings of shape (B, N, D)
        idx: Edge indices of shape (3, E), where the first row is batch indices
        batch_size: Number of graphs in the batch
        """
        if idx.numel() == 0:
            return h.new_empty((0,))

        num_edges = idx.size(1)

        if batch_index is None:
            batch_index = torch.zeros(num_edges, device=h.device, dtype=torch.long)
        else:
            batch_index = batch_index.to(h.device).long()
            if batch_index.numel() != num_edges:
                batch_index = batch_index[:num_edges]

        idx = idx.to(h.device).long()

        emb_in = h[batch_index, idx[0], :]
        emb_out = h[batch_index, idx[1], :]

        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)

        scores = self.fd_decoder.forward(sqdist)
        scores = torch.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
        scores = scores.clamp(min=1e-6, max=1.0 - 1e-6)

        return scores
    
    def sample_edges(self, edge_index, num_nodes=None):
        """
        Convert dense adjacency matrix to sparse edges and sample negative edges.
        edge_index: Dense adjacency matrix of shape (B, N, N)
        num_nodes: Number of nodes per graph
        """
        batch_size, N, _ = edge_index.size()

        pos_edges = []
        neg_edges = []
        batch_idx = []
        for i in range(batch_size):
            pos = torch.nonzero(edge_index[i], as_tuple=False)  # shape: (E_i, 2)
            num_pos = pos.size(0)
            if num_pos == 0:
                continue
                
            row = torch.arange(N, device=edge_index.device).repeat_interleave(N)
            col = torch.arange(N, device=edge_index.device).repeat(N)
            all_edges = torch.stack([row, col], dim=-1)  # shape: (N*N, 2)

            pos_ids = pos[:, 0] * N + pos[:, 1]
            all_ids = all_edges[:, 0] * N + all_edges[:, 1]
            
            pos_ids_set = set(pos_ids.tolist())
            mask = [(id_ not in pos_ids_set) for id_ in all_ids.tolist()]
            mask = torch.tensor(mask, dtype=torch.bool, device=edge_index.device)
            neg_candidates = all_edges[mask]  # (N*N - E_i, 2)

            if neg_candidates.size(0) < num_pos:
                sampled_neg = neg_candidates
            else:
                rand_idx = torch.randperm(neg_candidates.size(0), device=edge_index.device)[:num_pos]
                sampled_neg = neg_candidates[rand_idx]
            
            pos_edges.append(pos)
            neg_edges.append(sampled_neg)
            batch_idx.append(torch.full((num_pos,), i, device=edge_index.device))

        if len(pos_edges) == 0:
            return (torch.empty((2,0), dtype=torch.long, device=edge_index.device),
                    torch.empty((2,0), dtype=torch.long, device=edge_index.device),
                    torch.empty((0,),    dtype=torch.long, device=edge_index.device))
        
        pos_edge_index = torch.cat(pos_edges, dim=0).t()  # shape: (2, E_pos)
        neg_edge_index = torch.cat(neg_edges, dim=0).t()  # shape: (2, E_neg)
        batch_index = torch.cat(batch_idx, dim=0)         # shape: (E_pos,)

        return pos_edge_index, neg_edge_index, batch_index
    
    def training_step(self, batch, batch_idx):
        dense_data, node_mask = utils.to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        x, E = dense_data.X, dense_data.E

        adj, edge_labels = utils.process_edge_attr(E, node_mask)
        node_labels = x
        # calculate the degree of each node as node feature
        degrees = adj.sum(dim=1)  # Summing over the rows gives the degree of each node
        x = torch.cat([x, degrees.unsqueeze(-1)], dim=-1)  

        
        z = self.encoder.encode(x, adj)
        
        pos_edge_index, neg_edge_index, batch_index = self.sample_edges(adj, x.size(0))
        
        pos_scores = self.edge_decode(z, pos_edge_index, z.size(0), batch_index)
        neg_scores = self.edge_decode(z, neg_edge_index, z.size(0), batch_index)
        
        pred_node_labels = self.nc_decoder.decode(z, adj)
        pred_edge_labels = self.lp_decoder.decode(z, adj)
        # pdb.set_trace()
        # pdb.set_trace()
        loss = self.train_loss.forward({'node':node_labels, 'edge':edge_labels}, {'node':pred_node_labels, 'edge':pred_edge_labels}, pos_scores, neg_scores, log=True)
        
        
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.cpu().numpy()) + list(neg_scores.data.cpu().numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        
        print("Validation metrics of edge:", metrics)
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
        dense_data, node_mask = utils.to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        x, E = dense_data.X, dense_data.E
        adj, edge_labels = utils.process_edge_attr(E, node_mask)
        node_labels = x
        # calculate the degree of each node as node feature
        degrees = adj.sum(dim=1)  # Summing over the rows gives the degree of each node
        x = torch.cat([x, degrees.unsqueeze(-1)], dim=-1)  
        
        z = self.encoder.encode(x, adj.squeeze())
        
        pos_edge_index, neg_edge_index, batch_index = self.sample_edges(adj, x.size(0))
        
        pos_scores = self.edge_decode(z, pos_edge_index, z.size(0), batch_index)
        neg_scores = self.edge_decode(z, neg_edge_index, z.size(0), batch_index)
        
        pred_node_labels = self.nc_decoder.decode(z, adj)
        pred_edge_labels = self.lp_decoder.decode(z, adj)
        
                
        loss = self.val_loss.forward({'node':node_labels, 'edge':edge_labels}, {'node':pred_node_labels, 'edge':pred_edge_labels}, pos_scores, neg_scores, log=True)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        
        return {'loss':loss}

    
    def test_step(self, batch, batch_idx):
        dense_data, node_mask = utils.to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        # dense_data = dense_data.mask(node_mask)
        x, E = dense_data.X, dense_data.E
        adj, edge_labels = utils.process_edge_attr(E, node_mask)
        node_labels = x
        
        # calculate the degree of each node as node feature
        degrees = adj.sum(dim=1)  # Summing over the rows gives the degree of each node
        x = torch.cat([x, degrees.unsqueeze(-1)], dim=-1)  

        z = self.encoder.encode(x, adj.squeeze())
        
        pos_edge_index, neg_edge_index, batch_index = self.sample_edges(adj, x.size(0))
        
        pos_scores = self.edge_decode(z, pos_edge_index, z.size(0), batch_index)
        neg_scores = self.edge_decode(z, neg_edge_index, z.size(0), batch_index)
           
        pred_node_labels = self.nc_decoder.decode(z, adj)
        pred_edge_labels = self.lp_decoder.decode(z, adj)
        
        loss = self.test_loss.forward({'node':node_labels, 'edge':edge_labels}, {'node':pred_node_labels, 'edge':pred_edge_labels}, pos_scores, neg_scores, log=True)
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        
        print("Validation metrics of edge:", metrics)
        
        ## compute node classification metrics
        node_labels = torch.argmax(node_labels, dim=-1)
        node_labels = node_labels.data.cpu().numpy()
        pred_node_labels = torch.argmax(pred_node_labels, dim=-1)
        pred_node_labels = pred_node_labels.data.cpu().numpy()
        node_acc = np.mean(node_labels == pred_node_labels)
        node_f1 = f1_score(node_labels.reshape(-1), pred_node_labels.reshape(-1), average='micro')
        node_metrics = {'acc': node_acc, 'f1': node_f1}
        
        print("Validation metrics of node:", node_metrics)
        
        # compute edge classification metrics
        edge_labels = torch.argmax(edge_labels, dim=-1)
        edge_labels = edge_labels.data.cpu().numpy()
        
        pred_edge_labels = torch.argmax(pred_edge_labels, dim=-1)
        pred_edge_labels = pred_edge_labels.data.cpu().numpy()
        edge_acc = np.mean(edge_labels == pred_edge_labels)
        edge_f1 = f1_score(edge_labels.reshape(-1), pred_edge_labels.reshape(-1), average='micro')
        edge_metrics = {'acc': edge_acc, 'f1': edge_f1}
        
        print("Validation metrics of edge:", edge_metrics)
        
        return {'loss':loss}
    
        
    
    
    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print("Starting training with manifold:", self.manifold_name)

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: epoch_edge_existent_bce: {to_log['train_epoch/epoch_edge_existent_bce'] :.3f}"
                      f" -- epoch_node_ce: {to_log['train_epoch/epoch_node_ce'] :.3f} --"
                      f" epoch_edge_ce: {to_log['train_epoch/epoch_edge_ce'] :.3f}"
                      f" -- {time.time() - self.start_epoch_time:.1f}s ")
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}")
        if torch.cuda.is_available():
            print(torch.cuda.memory_summary())
    
    def on_validation_epoch_start(self):
        self.val_loss.reset()
        
    def on_test_epoch_start(self):
        self.test_loss.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)
        return optimizer
    
    
        
        
        
        