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
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                output_all_encoded_layers=True,
                                                )
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        return sequence_output

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        #!start BPRloss
        # seq_out = self.forward(input_ids)
        # seq_out = seq_out[:, -1, :]
        # pos_ids, neg_ids = answers, neg_answers

        # # [batch seq_len hidden_size]
        # pos_emb = self.item_embeddings(pos_ids)
        # neg_emb = self.item_embeddings(neg_ids)

        # # [batch hidden_size]
        # seq_emb = seq_out # [batch*seq_len hidden_size]
        # pos_logits = torch.sum(pos_emb * seq_emb, -1) # [batch*seq_len]
        # neg_logits = torch.sum(neg_emb * seq_emb, -1)

        # loss = torch.mean(
        #     - torch.log(torch.sigmoid(pos_logits) + 1e-24) -
        #     torch.log(1 - torch.sigmoid(neg_logits) + 1e-24)
        # )
        #! end BPRloss

        #!start cross-entropy loss
        # seq_output = self.forward(input_ids)
        # seq_output = seq_output[:, -1, :]

        # # cross-entropy loss
        # test_item_emb = self.item_embeddings.weight
        # logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        # loss = nn.CrossEntropyLoss()(logits, answers)

        #!end cross-entropy loss


        #! start binary cross-entropy loss

        seq_output = self.forward(input_ids)
        seq_output = seq_output[:, -1, :]

        pos_ids, neg_ids = answers, neg_answers
        pos_emb = self.item_embeddings(pos_ids)      # [batch, hidden]
        neg_emb = self.item_embeddings(neg_ids)      # [batch, hidden]

        # Sequence embedding
        seq_emb = seq_output                        # [batch, hidden]

        # Compute logits
        pos_logits = torch.sum(pos_emb * seq_emb, dim=-1)  # [batch]
        neg_logits = torch.sum(neg_emb * seq_emb, dim=-1)  # [batch]

        # Create labels
        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)

        # Mask out padding positions (where pos_id == 0)
        valid_idx = (pos_ids != 0)

        # Compute BCE loss on valid positions
        loss = self.bce_loss(pos_logits[valid_idx], pos_labels[valid_idx])
        loss += self.bce_loss(neg_logits[valid_idx], neg_labels[valid_idx])
        #! end binary cross-entropy loss


        return loss

class SFSRecEncoder(nn.Module):
    def __init__(self, args):
        super(SFSRecEncoder, self).__init__()
        self.args = args
        block = SFSRecBlock(args)

        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, output_all_encoded_layers=False):

        all_encoder_layers = [ hidden_states ]

        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states,)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states) # hidden_states => torch.Size([256, 50, 64])

        return all_encoder_layers

class SFSRecBlock(nn.Module):
    def __init__(self, args):
        super(SFSRecBlock, self).__init__()
        self.layer = SFSRecLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states):
        layer_output = self.layer(hidden_states)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output

class SFSRecLayer(nn.Module):
    def __init__(self, args):
        super(SFSRecLayer, self).__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, args.max_seq_length//2 + 1, args.hidden_size, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)

    def forward(self, input_tensor):
        # input_tensor: [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape

        mean = input_tensor.mean(dim=1, keepdim=True)        # [batch, 1, hidden]
        sequence_emb_fft = mean.expand(-1, seq_len, -1).contiguous()  # [batch, seq_len, hidden]

        # 出力処理
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = hidden_states + input_tensor
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states
