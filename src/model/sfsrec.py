import torch
import torch.nn as nn
import copy
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, FeedForward

    
class SFSRecModel(SequentialRecModel):
    def __init__(self, args):
        super(SFSRecModel, self).__init__(args)

        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.item_encoder = SFSRecEncoder(args)
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        sequence_emb = self.add_position_embedding(input_ids)
        attention_mask = (input_ids > 0).long()
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                attention_mask=attention_mask,
                                                output_all_encoded_layers=True)
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        return sequence_output

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):

        seq_output = self.forward(input_ids)
        seq_output = seq_output[:, -1, :]

        pos_ids, neg_ids = answers, neg_answers
        pos_emb = self.item_embeddings(pos_ids)
        neg_emb = self.item_embeddings(neg_ids)

        seq_emb = seq_output

        pos_logits = torch.sum(pos_emb * seq_emb, dim=-1)
        neg_logits = torch.sum(neg_emb * seq_emb, dim=-1)

        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)

        valid_idx = (pos_ids != 0)

        loss = self.bce_loss(pos_logits[valid_idx], pos_labels[valid_idx])
        loss += self.bce_loss(neg_logits[valid_idx], neg_labels[valid_idx])


        return loss

class SFSRecEncoder(nn.Module):
    def __init__(self, args):
        super(SFSRecEncoder, self).__init__()
        self.args = args
        block = SFSRecBlock(args)

        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, output_all_encoded_layers=False):
        all_encoder_layers = [hidden_states]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask=attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class SFSRecBlock(nn.Module):
    def __init__(self, args):
        super(SFSRecBlock, self).__init__()
        self.layer = SFSRecLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask=None):
        layer_output = self.layer(hidden_states, attention_mask=attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output

class SFSRecLayer(nn.Module):
    def __init__(self, args):
        super(SFSRecLayer, self).__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, args.max_seq_length//2 + 1, args.hidden_size, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)

    def forward(self, input_tensor, attention_mask=None):

        batch, seq_len, hidden = input_tensor.shape
        if attention_mask is None:
            mean = input_tensor.mean(dim=1, keepdim=True)
        else:
            mask = attention_mask.unsqueeze(-1).float()
            masked_sum = torch.sum(input_tensor * mask, dim=1, keepdim=True)
            valid_counts = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
            mean = masked_sum / valid_counts

        sequence_emb = mean.expand(-1, seq_len, -1).contiguous()
        hidden_states = self.out_dropout(sequence_emb) + input_tensor
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states
