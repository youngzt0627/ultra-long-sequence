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
class ETADataset(Dataset):
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

class ETADataLoaders:
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
        train_ds = ETADataset(train_df, short_seq_dict, long_seq_dict, description, encoders, device, max_len, max_len_long)
        test_ds = ETADataset(test_df, short_seq_dict, long_seq_dict, description, encoders, device, max_len, max_len_long)
        
        self.train_loader = DataLoader(train_ds, batch_size=bsz, shuffle=True)
        self.test_loader = DataLoader(test_ds, batch_size=bsz, shuffle=False)

    def _load_seq(self, path):
        d = {}
        with open(path, 'r') as f:
            for line in f:
                uid, items = line.strip().split(':')
                d[uid] = items.split(',')
        return d

class EfficientTargetAttention(nn.Module):
    def __init__(self, num_units, num_heads=8, topk=20, n_hashes=3):
        super(EfficientTargetAttention, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.topk = topk
        self.n_hashes = n_hashes
        
        self.W_Q = nn.Linear(num_units, num_units)
        self.W_K = nn.Linear(num_units, num_units)
        self.W_V = nn.Linear(num_units, num_units)
        self.layer_norm = nn.LayerNorm(num_units // num_heads)
        random_matrix = torch.randn(num_units // num_heads, n_hashes)
        self.register_buffer('hash_codes', torch.sign(random_matrix))

    def encode_simhash_sign(self, inputs):
        return torch.matmul(inputs, self.hash_codes) > 0

    def forward(self, queries, keys, values, key_masks=None):
        B, T_q, C = queries.shape
        _, T_k, _ = keys.shape
        H = self.num_heads
        D = C // H

        Q = self.W_Q(queries) # [B, T_q, C]
        K = self.W_K(keys)    # [B, T_k, C]
        V = self.W_V(values)  # [B, T_k, C]

        Q_ = torch.cat(torch.split(Q, D, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, D, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, D, dim=2), dim=0)

        Q_ = self.layer_norm(Q_)
        K_ = self.layer_norm(K_)

        Q_sign = self.encode_simhash_sign(Q_) # [B*H, T_q, n_hashes]
        K_sign = self.encode_simhash_sign(K_) # [B*H, T_k, n_hashes]
        hamming_dist = (Q_sign.unsqueeze(2) ^ K_sign.unsqueeze(1)).sum(dim=-1).float()
        current_topk = min(self.topk, T_k)
        topk_values, topk_indices = torch.topk(-hamming_dist, current_topk, dim=-1)
        batch_idx = torch.arange(B * H, device=queries.device).unsqueeze(1)
        K_topk = K_[batch_idx, topk_indices.squeeze(1)]
        V_topk = V_[batch_idx, topk_indices.squeeze(1)]

        outputs = torch.matmul(Q_, K_topk.transpose(1, 2))
        outputs = outputs / (D ** 0.5)

        if key_masks is not None:
            key_masks_ = key_masks.repeat(H, 1)
            key_masks_topk = torch.gather(key_masks_, 1, topk_indices.squeeze(1))
            key_masks_topk = key_masks_topk.unsqueeze(1)
            paddings = torch.ones_like(outputs) * (-2**32 + 1)
            outputs = torch.where(key_masks_topk > 0, outputs, paddings)

        outputs = F.softmax(outputs, dim=-1)
        outputs = torch.matmul(outputs, V_topk)
        outputs = torch.cat(torch.split(outputs, B, dim=0), dim=2)
        return outputs.squeeze(1)

class TargetMultiHeadAttention(nn.Module):
    def __init__(self, num_units, num_heads=8):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.W_Q = nn.Linear(num_units, num_units)
        self.W_K = nn.Linear(num_units, num_units)
        self.W_V = nn.Linear(num_units, num_units)
        self.layer_norm = nn.LayerNorm(num_units // num_heads)

    def forward(self, queries, keys, values, key_masks=None):
        B, T_q, C = queries.shape
        _, T_k, _ = keys.shape
        H = self.num_heads
        D = C // H
        Q = self.W_Q(queries)
        K = self.W_K(keys)
        V = self.W_V(values)
        Q_ = torch.cat(torch.split(Q, D, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, D, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, D, dim=2), dim=0)
        Q_ = self.layer_norm(Q_)
        K_ = self.layer_norm(K_)
        scores = torch.matmul(Q_, K_.transpose(1, 2)) / (D ** 0.5)
        if key_masks is not None:
            key_masks_ = key_masks.repeat(H, 1)
            key_masks_ = key_masks_.unsqueeze(1)
            paddings = torch.ones_like(scores) * (-2**32 + 1)
            scores = torch.where(key_masks_ > 0, scores, paddings)
        weights = F.softmax(scores, dim=-1)
        outputs = torch.matmul(weights, V_)
        outputs = torch.cat(torch.split(outputs, B, dim=0), dim=2)
        return outputs.squeeze(1)

class ETAModel(nn.Module):
    def __init__(self, description, embed_dim=10, mlp_dims=(128, 64), dropout=0.2, num_heads=8, topk=20):
        super(ETAModel, self).__init__()
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
                # 兴趣向量维度：item(8) + cate(8) + brand(8) + cross_cate(1) + cross_brand(1) = 26
                input_dim += 3 * embed_dim + 2

        self.eta_attention = EfficientTargetAttention(
            num_units=3 * embed_dim + 2, 
            num_heads=num_heads, 
            topk=topk
        )
        self.short_attention = TargetMultiHeadAttention(num_units=3 * embed_dim + 2, num_heads=num_heads)
        
        self.mlp = MultiLayerPerceptron(input_dim, mlp_dims, dropout)
        
    def _get_size(self, name):
        return [size for n, size, t in self.description if n == name][0]

    def forward(self, x_dict):
        all_feats = []
        
        # 1. 基础特征提取
        target_item_emb = self.item_embed(x_dict['item_id'].squeeze(-1))
        target_cate_emb = self.cate_embed(x_dict['cate_id'].squeeze(-1))
        target_brand_emb = self.brand_embed(x_dict['brand'].squeeze(-1))
        
        # Target 向量：嵌入向量 + 交叉特征数值
        qi_full = torch.cat([
            target_item_emb, target_cate_emb, target_brand_emb,
            x_dict['u_cate_click_cnt'], x_dict['u_brand_click_cnt']
        ], dim=-1).unsqueeze(1)
        
        # 2. 长序列建模
        seq_item_emb = self.item_embed(x_dict['long_item_seq'])
        seq_cate_emb = self.cate_embed(x_dict['long_cate_seq'])
        seq_brand_emb = self.brand_embed(x_dict['long_brand_seq'])
        # 长序列向量：嵌入向量序列 + 交叉特征序列
        H_long = torch.cat([
            seq_item_emb, seq_cate_emb, seq_brand_emb,
            x_dict['long_u_cate_click_cnt_seq'].unsqueeze(-1),
            x_dict['long_u_brand_click_cnt_seq'].unsqueeze(-1)
        ], dim=-1)
        
        # 3. 处理分类与数值特征
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
        long_interest = self.eta_attention(queries=qi_full, keys=H_long, values=H_long, key_masks=mask_long)

        # 4. 短序列建模
        hist_item_emb = self.item_embed(x_dict['hist_item_seq'])
        hist_cate_emb = self.cate_embed(x_dict['hist_cate_seq'])
        hist_brand_emb = self.brand_embed(x_dict['hist_brand_seq'])
        # 短序列向量：嵌入向量序列 + 交叉特征序列
        H_hist = torch.cat([
            hist_item_emb, hist_cate_emb, hist_brand_emb,
            x_dict['hist_u_cate_click_cnt_seq'].unsqueeze(-1),
            x_dict['hist_u_brand_click_cnt_seq'].unsqueeze(-1)
        ], dim=-1)
        mask_hist = (x_dict['hist_item_seq'] > 0).float()
        short_interest = self.short_attention(qi_full, H_hist, H_hist, mask_hist)

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
    loaders = ETADataLoaders(args.data_dir, args.bsz, device)
    
    print(f"Building ETA Model (Heads: {args.num_heads}, TopK: {args.topk})...")
    model = ETAModel(loaders.description, num_heads=args.num_heads, topk=args.topk).to(device)
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
