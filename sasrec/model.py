import numpy as np
import torch
import torch.nn as nn
from utils import get_device  # Importing get_device from utils.py


device = get_device()
# Use 'device' wherever tensors are moved to a device, for example:
# tensor.to(device)


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1).to(device)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate).to(device)
        self.relu = torch.nn.ReLU().to(device)
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1).to(device)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate).to(device)

    def forward(self, inputs):
        inputs = inputs.to(device)  # Move inputs to the correct device
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = get_device()  # Use get_device to determine the correct device

        # Embeddings
        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0).to(self.dev)
        self.pos_emb = nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0).to(self.dev)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate).to(self.dev)

        # Transformer Layers
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8).to(self.dev)

        for _ in range(args.num_blocks):
            new_attn_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8).to(self.dev)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate).to(self.dev)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8).to(self.dev)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate).to(self.dev)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs):
        # Convert log_seqs to a tensor on the correct device
        if isinstance(log_seqs, np.ndarray):
            log_seqs = torch.tensor(log_seqs, dtype=torch.long, device=self.dev)
        else:
            log_seqs = log_seqs.to(self.dev)

        # Apply item and positional embeddings
        seqs = self.item_emb(log_seqs) * (self.item_emb.embedding_dim ** 0.5)
        poss = torch.arange(1, log_seqs.shape[1] + 1, device=self.dev).unsqueeze(0).repeat(log_seqs.shape[0], 1)
        poss = poss * (log_seqs != 0).long()
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]  # Time dimension length for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):  # for training
        # Ensure inputs are on the correct device
        if isinstance(user_ids, np.ndarray):
            user_ids = torch.tensor(user_ids, dtype=torch.long, device=self.dev)
        if isinstance(log_seqs, np.ndarray):
            log_seqs = torch.tensor(log_seqs, dtype=torch.long, device=self.dev)
        if isinstance(pos_seqs, np.ndarray):
            pos_seqs = torch.tensor(pos_seqs, dtype=torch.long, device=self.dev)
        if isinstance(neg_seqs, np.ndarray):
            neg_seqs = torch.tensor(neg_seqs, dtype=torch.long, device=self.dev)

        user_ids = user_ids.to(self.dev)
        log_seqs = log_seqs.to(self.dev)
        pos_seqs = pos_seqs.to(self.dev)
        neg_seqs = neg_seqs.to(self.dev)

        log_feats = self.log2feats(log_seqs)  # Generate features from log sequences

        pos_embs = self.item_emb(pos_seqs)  # Get positive embeddings
        neg_embs = self.item_emb(neg_seqs)  # Get negative embeddings

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        # Ensure inputs are on the correct device
        if isinstance(user_ids, np.ndarray):
            user_ids = torch.tensor(user_ids, dtype=torch.long, device=self.dev)
        if isinstance(log_seqs, np.ndarray):
            log_seqs = torch.tensor(log_seqs, dtype=torch.long, device=self.dev)
        if isinstance(item_indices, np.ndarray):
            item_indices = torch.tensor(item_indices, dtype=torch.long, device=self.dev)

        user_ids = user_ids.to(self.dev)
        log_seqs = log_seqs.to(self.dev)
        item_indices = item_indices.to(self.dev)

        log_feats = self.log2feats(log_seqs)  # Generate features from log sequences

        final_feat = log_feats[:, -1, :]  # Use only the last feature for prediction

        item_embs = self.item_emb(item_indices)  # Get item embeddings

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits