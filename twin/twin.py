import os
import torch
import numpy as np
import pandas as pd
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

# ================= MLP 组件 =================
class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

# ================= 数据加载与处理 =================
class TWINDataset(Dataset):
    """
    数据集类：将 DataFrame 和 预加载的序列 转换为 Tensor。
    """
    def __init__(self, df, short_seq_dict, long_seq_dict, description, encoders, device, max_len=20, max_len_long=None):
        self.df = df
        self.short_seq_dict = short_seq_dict
        self.long_seq_dict = long_seq_dict
        self.description = description
        self.encoders = encoders
        self.device = device
        self.max_len = max_len
        self.max_len_long = max_len_long or max_len
        
        # 分类特征、连续特征、序列特征、标签
        self.spr_names = [n for n, _, t in description if t == 'spr']
        self.ctn_names = [n for n, _, t in description if t == 'ctn']
        self.seq_names = [n for n, _, t in description if t == 'seq']
        self.seq_ctn_names = [n for n, _, t in description if t == 'seq_ctn']
        self.label_name = [n for n, _, t in description if t == 'label'][0]
        
        self._prepare_data()

    def _prepare_data(self):
        self.tensors = {}
        
        # 1. 处理离散特征
        for name in self.spr_names:
            enc = self.encoders[name]
            vals = [enc.get(str(v), 0) for v in self.df[name].tolist()]
            self.tensors[name] = torch.tensor(vals, dtype=torch.long, device=self.device).unsqueeze(-1)
            
        # 2. 处理连续特征
        for name in self.ctn_names:
            vals = self.df[name].values.astype(np.float32)
            self.tensors[name] = torch.tensor(vals, dtype=torch.float32, device=self.device).unsqueeze(-1)
            
        # 3. 处理序列特征 (离散)
        for name in self.seq_names:
            base_name = name.replace('hist_', '').replace('long_', '').replace('_seq', '')
            if base_name == 'item': base_name = 'item_id'
            elif base_name == 'cate': base_name = 'cate_id'
            
            enc = self.encoders[base_name]
            seq_list = []
            for uid in self.df['user_id'].astype(str).tolist():
                if name.startswith('hist_'):
                    src = self.short_seq_dict[name]
                    pad_len = self.max_len
                else:
                    src = self.long_seq_dict[name]
                    pad_len = self.max_len_long
                raw_seq = src.get(uid, [])
                idx_seq = [enc.get(str(iid), 0) for iid in raw_seq]

                if len(idx_seq) < pad_len:
                    idx_seq += [0] * (pad_len - len(idx_seq))
                else:
                    idx_seq = idx_seq[-pad_len:]
                seq_list.append(idx_seq)
            self.tensors[name] = torch.tensor(seq_list, dtype=torch.long, device=self.device)

        # 4. 处理序列特征 (连续 - 交叉特征序列)
        for name in self.seq_ctn_names:
            seq_list = []
            for uid in self.df['user_id'].astype(str).tolist():
                if name.startswith('hist_'):
                    src = self.short_seq_dict[name]
                    pad_len = self.max_len
                else:
                    src = self.long_seq_dict[name]
                    pad_len = self.max_len_long
                raw_seq = src.get(uid, [])
                val_seq = [float(v) for v in raw_seq]

                if len(val_seq) < pad_len:
                    val_seq += [0.0] * (pad_len - len(val_seq))
                else:
                    val_seq = val_seq[-pad_len:]
                seq_list.append(val_seq)
            self.tensors[name] = torch.tensor(seq_list, dtype=torch.float32, device=self.device)
            
        # 5. 处理标签
        print("Finalizing tensors...")
        self.tensors['label'] = torch.tensor(self.df['clk'].values, dtype=torch.float32, device=self.device).unsqueeze(-1)

    def __getitem__(self, index):
        feat = {name: self.tensors[name][index] for name in self.tensors if name != 'label'}
        label = self.tensors['label'][index]
        return feat, label

    def __len__(self):
        return len(self.df)

class TWINDataLoaders:
    def __init__(self, data_dir, bsz, device, max_len=20, sample_n=100000):
        # 1. 加载 CSV (使用划分好的 train_joined_logs 和 test_joined_logs)
        print(f"Reading CSVs (sampling {sample_n} if needed)...")
        train_df = pd.read_csv(os.path.join(data_dir, 'train_joined_logs.csv'), nrows=sample_n)
        test_df = pd.read_csv(os.path.join(data_dir, 'test_joined_logs.csv'), nrows=sample_n // 5)
        
        train_df.columns = [c.strip() for c in train_df.columns]
        test_df.columns = [c.strip() for c in test_df.columns]
        
        # 2. 加载序列字典 (itemId, cateId, brand) - 短序列 + 完整长序列
        print("Loading short and long sequence dictionaries (including cross features)...")
        short_seq_dict = {
            "hist_item_seq": self._load_seq(os.path.join(data_dir, 'shortUserId2itemId.txt')),
            "hist_cate_seq": self._load_seq(os.path.join(data_dir, 'shortUserId2cateId.txt')),
            "hist_brand_seq": self._load_seq(os.path.join(data_dir, 'shortUserId2brand.txt')),
            "hist_u_cate_click_cnt_seq": self._load_seq(os.path.join(data_dir, 'shortUserId2uCateClickCnt.txt')),
            "hist_u_brand_click_cnt_seq": self._load_seq(os.path.join(data_dir, 'shortUserId2uBrandClickCnt.txt'))
        }

        long_seq_dict = {
            "long_item_seq": self._load_seq(os.path.join(data_dir, 'userId2itemId.txt')),
            "long_cate_seq": self._load_seq(os.path.join(data_dir, 'userId2cateId.txt')),
            "long_brand_seq": self._load_seq(os.path.join(data_dir, 'userId2brand.txt')),
            "long_u_cate_click_cnt_seq": self._load_seq(os.path.join(data_dir, 'userId2uCateClickCnt.txt')),
            "long_u_brand_click_cnt_seq": self._load_seq(os.path.join(data_dir, 'userId2uBrandClickCnt.txt'))
        }

        # 计算完整长序列的最大长度
        uids = pd.concat([train_df['user_id'], test_df['user_id']]).astype(str).unique()
        max_len_long = 0
        for uid in uids:
            max_len_long = max(max_len_long, len(long_seq_dict["long_item_seq"].get(uid, [])))
        print(f"Max length of long sequences: {max_len_long}")
        
        # 3. 定义列
        cat_cols = ['user_id','item_id','cms_segid','cms_group_id','final_gender_code','age_level','pvalue_level','shopping_level','occupation','new_user_class_level','cate_id','campaign_id','customer','brand']
        # 增加交叉特征
        ctn_cols = ['price', 'u_cate_click_cnt', 'u_brand_click_cnt']
        
        # 连续特征归一化
        for c in ctn_cols:
            m, s = train_df[c].mean(), train_df[c].std() + 1e-9
            train_df[c] = (train_df[c] - m) / s
            test_df[c] = (test_df[c] - m) / s
            
        # 4. 构建编码器
        encoders = {}
        description = []
        for c in cat_cols:
            all_vals = pd.concat([train_df[c], test_df[c]]).astype(str).unique()
            enc = {v: i + 1 for i, v in enumerate(sorted(all_vals))} # 0 留给 padding
            encoders[c] = enc
            description.append((c, len(enc) + 1, 'spr'))
            
        for c in ctn_cols:
            description.append((c, 1, 'ctn'))
            
        # 增加短序列特征描述 (包含交叉特征序列)
        description.append(('hist_item_seq', encoders['item_id'].__len__() + 1, 'seq'))
        description.append(('hist_cate_seq', encoders['cate_id'].__len__() + 1, 'seq'))
        description.append(('hist_brand_seq', encoders['brand'].__len__() + 1, 'seq'))
        description.append(('hist_u_cate_click_cnt_seq', 1, 'seq_ctn'))
        description.append(('hist_u_brand_click_cnt_seq', 1, 'seq_ctn'))

        # 增加完整长序列特征描述 (包含交叉特征序列)
        description.append(('long_item_seq', encoders['item_id'].__len__() + 1, 'seq'))
        description.append(('long_cate_seq', encoders['cate_id'].__len__() + 1, 'seq'))
        description.append(('long_brand_seq', encoders['brand'].__len__() + 1, 'seq'))
        description.append(('long_u_cate_click_cnt_seq', 1, 'seq_ctn'))
        description.append(('long_u_brand_click_cnt_seq', 1, 'seq_ctn'))
        description.append(('clk', 1, 'label'))
        
        self.description = description
        self.encoders = encoders
        
        # 5. 创建 DataLoader
        train_ds = TWINDataset(train_df, short_seq_dict, long_seq_dict, description, encoders, device, max_len, max_len_long)
        test_ds = TWINDataset(test_df, short_seq_dict, long_seq_dict, description, encoders, device, max_len, max_len_long)
        
        self.train_loader = DataLoader(train_ds, batch_size=bsz, shuffle=True)
        self.test_loader = DataLoader(test_ds, batch_size=bsz, shuffle=False)

    def _load_seq(self, path):
        d = {}
        with open(path, 'r') as f:
            for line in f:
                uid, items = line.strip().split(':')
                d[uid] = items.split(',')
        return d

class TWINEfficientAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_units, num_heads=8, topk=20, cross_dim=2):
        super(TWINEfficientAttention, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.topk = topk
        self.W_Q = nn.Linear(query_dim, num_units)
        self.W_K = nn.Linear(key_dim, num_units)
        self.W_V = nn.Linear(value_dim, num_units)
        self.layer_norm = nn.LayerNorm(num_units // num_heads)
        
        # 交叉特征得分矩阵 W: 输入仅为 seq_cross(2)
        self.cross_score_net = nn.Linear(cross_dim, 1)

    def forward(self, queries, keys, values, seq_cross, key_masks=None):
        B, T_q, _ = queries.shape
        _, T_k, _ = keys.shape
        H = self.num_heads
        D = self.num_units // H

        # 1. QKV 投影 (Query/Value 包含交叉特征，Key 不包含)
        Q = self.W_Q(queries) # [B, T_q, num_units]
        K = self.W_K(keys)    # [B, T_k, num_units]
        V = self.W_V(values)  # [B, T_k, num_units]

        Q_ = torch.cat(torch.split(Q, D, dim=2), dim=0)   # [B*H, T_q, D]
        K_ = torch.cat(torch.split(K, D, dim=2), dim=0)   # [B*H, T_k, D]
        V_ = torch.cat(torch.split(V, D, dim=2), dim=0)   # [B*H, T_k, D]

        Q_ = self.layer_norm(Q_)
        K_ = self.layer_norm(K_)

        # 2. 计算 QKV 相似度分数
        scores_all = torch.matmul(Q_, K_.transpose(1, 2)) / (D ** 0.5)  # [B*H, T_q, T_k]

        # 3. 计算交叉特征得分 (仅基于历史行为的交叉特征)
        # seq_cross: [B, T_k, 2]
        cross_scores = self.cross_score_net(seq_cross).squeeze(-1)   # [B, T_k]
        cross_scores_all = cross_scores.repeat(H, 1).unsqueeze(1)      # [B*H, 1, T_k]

        # 4. 融合分数
        scores_all = scores_all + cross_scores_all

        if key_masks is not None:
            key_masks_ = key_masks.repeat(H, 1)                          # [B*H, T_k]
            scores_all = scores_all.masked_fill(key_masks_.unsqueeze(1) <= 0, -1e9)

        current_topk = min(self.topk, T_k)
        topk_vals, topk_idx = torch.topk(scores_all, current_topk, dim=-1)  # [B*H, T_q, K]

        batch_idx = torch.arange(B * H, device=queries.device).unsqueeze(1)
        V_topk = V_[batch_idx, topk_idx.squeeze(1)]  # [B*H, K, D]

        attn_weights = F.softmax(topk_vals, dim=-1)  # [B*H, T_q, K]
        outputs = torch.matmul(attn_weights, V_topk) # [B*H, T_q, D]

        outputs = torch.cat(torch.split(outputs, B, dim=0), dim=2)  # [B, T_q, num_units]
        return outputs.squeeze(1)

class TargetMultiHeadAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_units, num_heads=8, cross_dim=2):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.W_Q = nn.Linear(query_dim, num_units)
        self.W_K = nn.Linear(key_dim, num_units)
        self.W_V = nn.Linear(value_dim, num_units)
        self.layer_norm = nn.LayerNorm(num_units // num_heads)
        self.cross_score_net = nn.Linear(cross_dim, 1)

    def forward(self, queries, keys, values, seq_cross, key_masks=None):
        B, T_q, _ = queries.shape
        _, T_k, _ = keys.shape
        H = self.num_heads
        D = self.num_units // H
        
        Q = self.W_Q(queries)
        K = self.W_K(keys)
        V = self.W_V(values)
        Q_ = torch.cat(torch.split(Q, D, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, D, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, D, dim=2), dim=0)
        Q_ = self.layer_norm(Q_)
        K_ = self.layer_norm(K_)
        
        # QKV 分数
        scores = torch.matmul(Q_, K_.transpose(1, 2)) / (D ** 0.5) # [B*H, 1, T_k]
        
        # 交叉特征分数 (仅使用历史序列的交叉特征)
        cross_scores = self.cross_score_net(seq_cross).squeeze(-1) # [B, T_k]
        cross_scores_all = cross_scores.repeat(H, 1).unsqueeze(1)     # [B*H, 1, T_k]
        
        scores = scores + cross_scores_all

        if key_masks is not None:
            key_masks_ = key_masks.repeat(H, 1)
            key_masks_ = key_masks_.unsqueeze(1)
            paddings = torch.ones_like(scores) * (-2**32 + 1)
            scores = torch.where(key_masks_ > 0, scores, paddings)
        weights = F.softmax(scores, dim=-1)
        outputs = torch.matmul(weights, V_)
        outputs = torch.cat(torch.split(outputs, B, dim=0), dim=2)
        return outputs.squeeze(1)

class TWINModel(nn.Module):
    def __init__(self, description, embed_dim=10, mlp_dims=(128, 64), dropout=0.2, num_heads=8, topk=20):
        super(TWINModel, self).__init__()
        self.description = description
        
        self.item_embed = nn.Embedding(self._get_size('item_id'), embed_dim, padding_idx=0)
        self.cate_embed = nn.Embedding(self._get_size('cate_id'), embed_dim, padding_idx=0)
        self.brand_embed = nn.Embedding(self._get_size('brand'), embed_dim, padding_idx=0)
        
        self.embed_layers = nn.ModuleDict()
        input_dim = 0
        for name, size, t in description:
            if t == 'spr':
                if name in ['item_id', 'cate_id', 'brand']:
                    input_dim += embed_dim
                else:
                    self.embed_layers[name] = nn.Embedding(size, embed_dim, padding_idx=0)
                    input_dim += embed_dim
            elif t == 'ctn':
                input_dim += 1
            elif t == 'seq' and name in ('hist_item_seq', 'long_item_seq'):
                # 兴趣向量维度：由 num_units (32) 决定
                input_dim += 32

        # 定义维度
        query_dim = 3 * embed_dim + 2 # 30 + 2 = 32
        key_dim = 3 * embed_dim       # 30
        value_dim = 3 * embed_dim + 2 # 30 + 2 = 32
        num_units = 32                # 必须能被 num_heads(8) 整除

        self.twin_attention = TWINEfficientAttention(
            query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, 
            num_units=num_units, num_heads=num_heads, topk=topk
        )
        self.short_attention = TargetMultiHeadAttention(
            query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, 
            num_units=num_units, num_heads=num_heads
        )
        
        self.mlp = MultiLayerPerceptron(input_dim, mlp_dims, dropout)
        
    def _get_size(self, name):
        return [size for n, size, t in self.description if n == name][0]

    def forward(self, x_dict):
        all_feats = []
        
        # 1. 基础特征提取
        target_item_emb = self.item_embed(x_dict['item_id'].squeeze(-1))
        target_cate_emb = self.cate_embed(x_dict['cate_id'].squeeze(-1))
        target_brand_emb = self.brand_embed(x_dict['brand'].squeeze(-1))
        
        # Query 向量：Embedding (30) + 交叉特征 (2) = 32
        qi_full = torch.cat([
            target_item_emb, target_cate_emb, target_brand_emb,
            x_dict['u_cate_click_cnt'], x_dict['u_brand_click_cnt']
        ], dim=-1).unsqueeze(1)
        
        # 2. 长序列建模
        seq_item_emb = self.item_embed(x_dict['long_item_seq'])
        seq_cate_emb = self.cate_embed(x_dict['long_cate_seq'])
        seq_brand_emb = self.brand_embed(x_dict['long_brand_seq'])
        
        # Key 向量 (仅 Embedding): 30
        H_long_emb = torch.cat([seq_item_emb, seq_cate_emb, seq_brand_emb], dim=-1)
        
        # 长序列交叉特征
        H_long_cross = torch.cat([
            x_dict['long_u_cate_click_cnt_seq'].unsqueeze(-1),
            x_dict['long_u_brand_click_cnt_seq'].unsqueeze(-1)
        ], dim=-1)
        
        # Value 向量 (Embedding + 交叉特征): 32
        H_long_full = torch.cat([H_long_emb, H_long_cross], dim=-1)
        
        # 3. 处理分类与数值特征 (拼接进 MLP)
        for name, _, t in self.description:
            if t == 'spr':
                x = x_dict[name].squeeze(-1)
                if name == 'item_id':
                    all_feats.append(target_item_emb)
                elif name == 'cate_id':
                    all_feats.append(target_cate_emb)
                elif name == 'brand':
                    all_feats.append(target_brand_emb)
                else:
                    all_feats.append(self.embed_layers[name](x))
            elif t == 'ctn':
                all_feats.append(x_dict[name])

        mask_long = (x_dict['long_item_seq'] > 0).float()
        long_interest = self.twin_attention(
            queries=qi_full, keys=H_long_emb, values=H_long_full, 
            seq_cross=H_long_cross, key_masks=mask_long
        )

        # 4. 短序列建模
        hist_item_emb = self.item_embed(x_dict['hist_item_seq'])
        hist_cate_emb = self.cate_embed(x_dict['hist_cate_seq'])
        hist_brand_emb = self.brand_embed(x_dict['hist_brand_seq'])
        
        # 短序列向量 (仅 Embedding): 30
        H_hist_emb = torch.cat([hist_item_emb, hist_cate_emb, hist_brand_emb], dim=-1)
        
        # 短序列交叉特征
        H_hist_cross = torch.cat([
            x_dict['hist_u_cate_click_cnt_seq'].unsqueeze(-1),
            x_dict['hist_u_brand_click_cnt_seq'].unsqueeze(-1)
        ], dim=-1)
        
        # 短序列 Value 向量 (Embedding + 交叉特征): 32
        H_hist_full = torch.cat([H_hist_emb, H_hist_cross], dim=-1)
        
        mask_hist = (x_dict['hist_item_seq'] > 0).float()
        short_interest = self.short_attention(
            qi_full, H_hist_emb, H_hist_full, 
            seq_cross=H_hist_cross, key_masks=mask_hist
        )

        all_feats.append(long_interest)
        all_feats.append(short_interest)
        final_input = torch.cat(all_feats, dim=1)
        out = torch.sigmoid(self.mlp(final_input).squeeze(-1))
        return out

# ================= 训练与评估 =================
def train_eval():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--bsz', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--topk', type=int, default=20)
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Loading data from {args.data_dir}...")
    loaders = TWINDataLoaders(args.data_dir, args.bsz, device)
    
    print(f"Building TWIN Model (Heads: {args.num_heads}, TopK: {args.topk})...")
    model = TWINModel(loaders.description, num_heads=args.num_heads, topk=args.topk).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        for i, (feat, label) in enumerate(loaders.train_loader):
            optimizer.zero_grad()
            pred = model(feat)
            loss = criterion(pred, label.squeeze(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i+1) % 100 == 0:
                print(f"Epoch {epoch+1} | Iter {i+1} | Loss {loss.item():.4f}")
        
        # Test
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for feat, label in loaders.test_loader:
                pred = model(feat)
                preds.extend(pred.tolist())
                labels.extend(label.tolist())
        auc = roc_auc_score(labels, preds)
        print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(loaders.train_loader):.4f} | Test AUC: {auc:.4f}")

if __name__ == '__main__':
    train_eval()
