# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

from models.general.LightGCN import LightGCN  #导入LightGCL
from models.BaseModel import GeneralModel
from models.BaseImpressionModel import ImpressionModel



class XSimGCL(LightGCN):
	reader = 'BaseReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size', 'n_layers', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = LightGCN.parse_model_args(parser)
		parser.add_argument('--reg_weight', type=float, default=1e-5,
							help='Weight of embedding regularization.')
		parser.add_argument('--eps', type=float, default=0.2,
							help='Magnitude of noise for embedding augmentation.')
		parser.add_argument('--tau', type=float, default=0.1,
							help='Temperature for contrastive loss.')
		return parser


	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.reg_weight = args.reg_weight
		self.eps = args.eps
		self.tau = args.tau


	def _sample_noise(self, shape):
		noise = torch.randn(shape, device=self.device)
		noise = F.normalize(noise, dim=1)  # L2 归一化
		sign = torch.randint(0, 2, (shape[0],), device=self.device).float().unsqueeze(1) * 2.0 - 1.0
		return sign * self.eps * noise

	# src/models/general/XSimGCL.py
	# vvv  请添加这个全新的辅助函数 vvv
	def _propagate(self, user_emb, item_emb):
		# 这是 LightGCN 的核心传播逻辑
		# 我们从 self.encoder 访问 LightGCN 的参数
		ego_embeddings = torch.cat([user_emb, item_emb], 0)
		all_embeddings = [ego_embeddings]
		
		adj_mat = self.encoder.sparse_norm_adj
		n_layers = len(self.encoder.layers)
		user_count = self.encoder.user_count
		item_count = self.encoder.item_count

		for k in range(n_layers):
			ego_embeddings = torch.sparse.mm(adj_mat, ego_embeddings)
			all_embeddings.append(ego_embeddings)
		
		all_embeddings = torch.stack(all_embeddings, 1)
		all_embeddings = torch.mean(all_embeddings, 1)
		
		user_all_embeddings, item_all_embeddings = torch.split(
			all_embeddings, [user_count, item_count])
		
		return user_all_embeddings, item_all_embeddings
	# ^^^  添加结束  ^^^
	# vvv  请用这段新代码替换你现有的整个 'forward' 方法 vvv
	def forward(self, feed_dict):
		self.check_list = []
		u_ids = feed_dict['user_id']
		i_ids = feed_dict['item_id']

		# 1. 获取原始的基础嵌入
		user_emb_base = self.encoder.embedding_dict['user_emb']
		item_emb_base = self.encoder.embedding_dict['item_emb']

		# 2. 创建视图 1 (带噪声)
		user_v1 = user_emb_base + self._sample_noise(user_emb_base.shape)
		item_v1 = item_emb_base + self._sample_noise(item_emb_base.shape)
		
		# 3. 创建视图 2 (带噪声)
		user_v2 = user_emb_base + self._sample_noise(user_emb_base.shape)
		item_v2 = item_emb_base + self._sample_noise(item_emb_base.shape)

		# 4. 对两个视图分别进行图传播 (使用我们刚添加的 _propagate)
		final_user_v1_all, final_item_v1_all = self._propagate(user_v1, item_v1)
		final_user_v2_all, final_item_v2_all = self._propagate(user_v2, item_v2)
		
		# 5. (重要!) 将两个视图的 *所有* 嵌入存入 feed_dict，以供 calculate_loss 使用
		feed_dict['user_embeds_v1'] = final_user_v1_all
		feed_dict['item_embeds_v1'] = final_item_v1_all
		feed_dict['user_embeds_v2'] = final_user_v2_all
		feed_dict['item_embeds_v2'] = final_item_v2_all
		
		# 6. 使用视图 1 的结果来进行 BPR 预测
		#    从 *所有* 嵌入中，取出当前 batch 需要的
		u_embeds_v1 = final_user_v1_all[u_ids]
		i_embeds_v1 = final_item_v1_all[i_ids]

		# 7. (这是你上次的修复) 处理评估阶段的维度匹配
		if i_ids.dim() > 1:
			u_embeds_v1 = u_embeds_v1.unsqueeze(1)
		
		prediction = (u_embeds_v1 * i_embeds_v1).sum(dim=-1)
		
		out_dict = {'prediction': prediction}
		return out_dict
	# ^^^  替换结束  ^^^

	def info_nce_loss(self, view1, view2, temperature):
		view1 = F.normalize(view1, dim=1)
		view2 = F.normalize(view2, dim=1)
		pos_score = (view1 * view2).sum(dim=-1)
		pos_score = torch.exp(pos_score / temperature)
		ttl_score = torch.matmul(view1, view2.transpose(0, 1))
		ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
		cl_loss = -torch.log(pos_score / ttl_score).mean()
		return cl_loss
	# ^^^ 添加这个辅助函数 ^^^

	# vvv 重写(Override) calculate_loss 方法 vvv
	def calculate_loss(self, feed_dict):
		# 1. 计算 BPR 推荐损失 (来自父类)
		bpr_loss = super().calculate_loss(feed_dict)

		# 2. 从 feed_dict 中取出两个视图的 embedding
		user_v1, item_v1 = feed_dict['user_embeds_v1'], feed_dict['item_embeds_v1']
		user_v2, item_v2 = feed_dict['user_embeds_v2'], feed_dict['item_embeds_v2']

		# 3. 获取 batch 中的 user 和 pos_item
		users = feed_dict['user_id']
		pos_items = feed_dict['pos_id']

		# 4. 计算对比损失 (InfoNCE)
		user_cl_loss = self.info_nce_loss(user_v1[users], user_v2[users], self.tau)
		item_cl_loss = self.info_nce_loss(item_v1[pos_items], item_v2[pos_items], self.tau)
		cl_loss = self.reg_weight * (user_cl_loss + item_cl_loss)

		# 5. 返回总损失
		return {'loss': bpr_loss + cl_loss}
	# ^^^ 重写(Override) calculate_loss 方法 ^^^