import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .vocabulary import EOS, PAD, SOS


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.gru = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.fp_proj = nn.Linear(1024, hidden_dim)#为每个gru层的初始状态提供不为0的相同向量

    def forward(self, src, src_lens, src_fp):
        emb = self.embedding(src)
        h0 = self.fp_proj(src_fp).unsqueeze(0).repeat(self.gru.num_layers, 1, 1)
        packed = pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        out, hidden = self.gru(packed, h0)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        return out, hidden


class Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_out, mask):
        src_len = encoder_out.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_out), dim=2)))
        scores = self.v(energy).squeeze(2)
        scores = scores.masked_fill(~mask, -1e9)
        return torch.softmax(scores, dim=1)


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.attention = Attention(hidden_dim)
        self.gru = nn.GRU(emb_dim + hidden_dim + hidden_dim, hidden_dim, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, hidden, encoder_out, src_mask, fp_proj):
        emb = self.dropout(self.embedding(input_ids))
        attn_weights = self.attention(hidden[-1], encoder_out, src_mask)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_out)
        fp_feat = fp_proj.unsqueeze(1).repeat(1, input_ids.size(1), 1)
        rnn_input = torch.cat((emb, context, fp_feat), dim=2)
        output, hidden = self.gru(rnn_input, hidden)
        pred = self.fc(torch.cat((output, context), dim=2))
        return pred, hidden, attn_weights


class ConditionalSeq2Seq(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.encoder = Encoder(vocab_size, emb_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(vocab_size, emb_dim, hidden_dim, num_layers, dropout)
        self.sim_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.qed_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, src, src_lens, tgt, src_fp, teacher_forcing_ratio: float):
        enc_out, hidden = self.encoder(src, src_lens, src_fp)
        fp_proj = self.encoder.fp_proj(src_fp)
        hidden = hidden + fp_proj.unsqueeze(0).repeat(hidden.size(0), 1, 1)
        src_mask = src != PAD
        batch_size, tgt_len = tgt.shape
        vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, tgt_len - 1, vocab_size, device=tgt.device)
        last_hidden = hidden
        dec_input = tgt[:, :1]
        for t in range(1, tgt_len):
            logits, hidden, _ = self.decoder(dec_input, hidden, enc_out, src_mask, fp_proj)
            outputs[:, t - 1] = logits[:, 0, :]
            last_hidden = hidden
            dec_input = tgt[:, t : t + 1] if random.random() < teacher_forcing_ratio else logits.argmax(dim=-1)
        dec_last_hidden = last_hidden[-1]#循环结束后再进行！sim和qed的预测，使用最后一个时间步最后一个gru层的dec hidden状态
        enc_mean = enc_out.mean(dim=1)
        sim_input = torch.cat([enc_mean, dec_last_hidden], dim=1)
        sim_pred = self.sim_head(sim_input).squeeze(-1)#只用于训练阶段
        qed_pred = self.qed_head(sim_input).squeeze(-1)#只用于训练阶段
        return outputs, sim_pred, qed_pred

    @torch.no_grad()
    def generate(self, src, src_lens, src_fp, max_len: int):#greedy coding 取 argmax最大的作为输出 
        self.eval()
        enc_out, hidden = self.encoder(src, src_lens, src_fp)
        fp_proj = self.encoder.fp_proj(src_fp)
        hidden = hidden + fp_proj.unsqueeze(0).repeat(hidden.size(0), 1, 1)
        src_mask = src != PAD
        batch_size = src.size(0)
        dec_input = torch.full((batch_size, 1), SOS, dtype=torch.long, device=src.device)
        generated = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            logits, hidden, _ = self.decoder(dec_input, hidden, enc_out, src_mask, fp_proj)
            next_id = logits.argmax(dim=-1)
            generated.append(next_id)
            finished |= next_id.squeeze(1).eq(EOS)
            dec_input = next_id
            if bool(finished.all()):
                break
        return torch.cat(generated, dim=1) if generated else torch.empty(batch_size, 0, dtype=torch.long, device=src.device)

    @torch.no_grad()
    def sample(self, src, src_lens, src_fp, max_len: int, temperature: float, top_k: int):#random sampling 得到的输出仍然是随机的
        self.eval()
        enc_out, hidden = self.encoder(src, src_lens, src_fp)
        fp_proj = self.encoder.fp_proj(src_fp)
        hidden = hidden + fp_proj.unsqueeze(0).repeat(hidden.size(0), 1, 1)
        src_mask = src != PAD
        batch_size = src.size(0)
        dec_input = torch.full((batch_size, 1), SOS, dtype=torch.long, device=src.device)
        generated = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            logits, hidden, _ = self.decoder(dec_input, hidden, enc_out, src_mask, fp_proj)#原始logits
            step_logits = logits[:, 0, :] / max(temperature, 1e-5)#除以温度的logits并进入循环
            if top_k > 0 and top_k < step_logits.size(-1):#筛选出top-k后需要再次进行softmax
                topv, topi = torch.topk(step_logits, k=top_k, dim=-1)#topv是top-k的logits，topi是对应的token id
                probs = torch.softmax(topv, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1)#随机采样的关键函数！
                next_id = torch.gather(topi, 1, sampled)#将softxmax结果的索引还原为token id
            else:
                probs = torch.softmax(step_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            generated.append(next_id)
            finished |= next_id.squeeze(1).eq(EOS)
            dec_input = next_id
            if bool(finished.all()):
                break
        return torch.cat(generated, dim=1) if generated else torch.empty(batch_size, 0, dtype=torch.long, device=src.device)
