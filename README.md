# QED 分子优化项目（GRU + Attention）
输入一个分子的 `SELFIES`，模型生成结构相似但 `QED` 更高的候选分子。
## 1. 关键指标解释

- `Validity`：生成分子是否可被 RDKit 解析的比例，越高越好。  
- `SimSrc`：生成分子与输入分子的相似度（Tanimoto）均值，越高表示结构改动越小。  
- `SimTgt`：生成分子与训练目标分子的相似度均值。  
- `dQED`：`QED(gen) - QED(src)` 的均值，大于 0 表示整体优化有效。  
- `positive_dqed_rate`：`dQED > 0` 的样本比例。  
- `opt30_rate`：满足 `SimSrc >= 0.30 且 dQED > 0` 的比例。  
- `opt40_rate`：满足 `SimSrc >= 0.40 且 dQED > 0` 的比例。  

## 2. 数据集处理流程

`qed_project/data.py`：

- 读取输入文件  
- 清洗分子：补全 SELFIES、token 化、RDKit 解析、计算 QED、提取 Morgan 指纹。  
- 构造训练 pair（source -> target）：  
  - `target_qed - source_qed >= min_pair_dqed`  
  - `pair_similarity >= min_pair_similarity`  
  - 支持重原子数差与 token 长度差约束  
  - 每个 source 按 `pair_score = sim*2 + dQED*5` 选 `top_k`  
- 将 pairs 划分为 train/valid/test。  

### 2.1 数据划分：


1. 读取并标准化数据列  
- 读入原始分子表，列名统一小写。  
- 至少要求 `smiles/selfies/qed` 三列。

2. 单分子预处理  
- 若 `selfies` 缺失但有 `smiles`，先做 `smiles -> selfies` 转换。  
- 将 `selfies` 拆为 token；无效样本跳过。  
- 用 RDKit 解析并标准化成 canonical SMILES。  
- 计算分子 QED、Morgan 指纹（1024 维）和 heavy atoms 数。

3. 选择 source 分子  
- 按 QED 从低到高排序，优先低 QED 分子作为 source。  
- 如果 source 太多，用 `max_src_samples` 随机采样控制计算量。

4. 粗筛 target 候选  
- 先按 heavy atoms 接近约束筛选（`max_heavy_atom_delta`）。  
- 目标分子必须满足 `target_qed - source_qed >= min_pair_dqed`。  
- 候选过多时按 `candidates_per_src` 随机下采样。

5. 精筛 + 打分形成 pair  
- 计算 source 与每个候选的 Tanimoto 相似度。  
- 过滤：  
  - `sim >= min_pair_similarity`  
  - 不能与 source 相同  
  - token 长度差不超过 `max_token_length_delta`  
- 计算 `pair_score = sim * 2.0 + dQED * 5.0`，每个 source 取 top-k target。  
- 汇总全部 source 的 top-k 结果，得到最终 pair 集合。

### 2.2 划分训练/验证/测试集

- 划分函数：`split_train_valid_test()`  
- 划分前先随机打乱（`seed=42`）  
- 划分比例（配置默认）：  
  - `train_ratio = 0.8`  
  - `valid_ratio = 0.1`  
  - `test_ratio = 0.1`  

即按 **8:1:1** 划分。

### 2.3 当前项目三个集合的具体 pair 数量


- 总 pair 数：`15560`  
- 训练集：`12448`  
- 验证集：`1556`  
- 测试集：`1556`  

## 3. 超参数设置（学习率 / Adam / 梯度裁剪 / 神经网络）

- 优化器：`Adam(lr=5e-4, weight_decay=1e-5)`  
- 学习率调度：`ReduceLROnPlateau(factor=0.5, patience=20)`  
- 梯度裁剪：`clip_grad_norm_(..., 1.0)`  
- 训练轮数：`epochs=80`  
- 批大小：`batch_size=32`  
- Teacher forcing：`0.6`  
- 模型参数：`emb_dim=128`, `hidden_dim=512`, `num_layers=3`, `dropout=0.3`  
- 多任务损失权重：`sim_loss_weight=1.0`, `qed_loss_weight=3.0`

总损失：

`Loss = CE + sim_loss_weight * MSE(sim_pred, pair_sim) + qed_loss_weight * MSE(qed_pred, tgt_qed)`

## 4. `qed_project` 文件介绍

- `config.py`：训练/推理超参数配置。  
- `data.py`：数据读取、pair 构造、Dataset/Collate。

- `model.py`：  
  - Encoder：Embedding + GRU，结合分子指纹初始化隐状态。  
  - Attention：解码时对编码器输出做注意力加权。  
  - Decoder：输入 token + context + 指纹特征，输出 token logits。  
  - `sim_head` / `qed_head`：辅助回归头，用于多任务学习。 

- `training.py`：训练主流程、保存模型、评估与可视化。  
- `evaluation.py`：验证/测试评估与画图。  
- `inference.py`：单分子预测与 rerank。  
- `metrics.py`：指标计算。  
- `chemistry.py`：RDKit/SELFIES 工具函数。  
- `vocabulary.py`：词表构建、编码解码。  
- `file_utils.py`：文件读写。  

## 5.`train.py` 和 `predict_one.py` 介绍

- `main/train.py`  
  - 训练入口脚本。  
  - 解析命令行参数（如 `data_path/out_dir/epochs/lr`）。  
  - 构建 `TrainConfig` 并调用 `qed_project.train(cfg)`。

- `main/predict_one.py`  
  - 单分子推理入口。  
  - 输入 `--input_selfies`，可选 `--checkpoint`。  
  - 调用 `qed_project.predict_one(cfg)`，输出优化后的 SELFIES/SMILES、相似度、QED、dQED。
## 6.`results`收录的结果
- 测试集与验证集评估
  - `test set's evaluation.csv`：测试集每条样本的预测与指标明细，便于逐条排查。
  - `validation set's evaluation.csv`：验证集每条样本的预测与指标明细，用于调参和误差分析。

- 评估汇总（整体指标）
  -`test set's metrics.json`：测试集总体指标汇总（如 validity、dQED、SimSrc、positive_dqed_rate 等），用于最终效果报告。
  -`validation set's metrics.json`：验证集总体指标汇总，用于模型选择和训练过程监控。

-可视化图表
  -`training_curves.png`：训练过程曲线（通常是 loss/metric 随 epoch 变化），用于判断收敛和过拟合。
  -`test_dqed_hist.png`：测试集 dQED 分布直方图，观察优化增益分布。
  -`test_simsrc_hist.png`：测试集 SimSrc 分布直方图，观察与原分子相似度分布。
  -`test_simsrc_dqed_scatter.png`：SimSrc 与 dQED 的散点关系图，用于分析“相似性-优化幅度”权衡。

-训练日志
  -`training log(detailed).csv`：详细训练日志（通常含每轮/每步 loss 与指标），用于复现实验和诊断训练异常。

-模型权重
  -`best model.pt`：验证表现最优的模型参数，通常用于最终推理/部署。
  -`last epoch's model.pt`：最后一轮模型参数，便于继续训练或对比“最佳 vs 最后”。

`main/predict_one.py`生成案例：
指令：
& C:/Users/16544/anaconda3/envs/my-pytorch/python.exe predict_one.py --input_selfies "[C][C][=N][N][Branch1][Branch2][C][=Branch1][C][=O][N][Ring1][=Branch1][C][=C][C][=C][C][=C][Ring1][=Branch1][F]" --out_dir output_sim_pairs

results：
输入 SELFIES: [C][C][=N][N][Branch1][Branch2][C][=Branch1][C][=O][N][Ring1][=Branch1][C][=C][C][=C][C][=C][Ring1][=Branch1][F]
优化后 SELFIES: [C][C][=N][N][Branch1][C][C][C][Branch1][C][C][=C][Ring1][#Branch1][C@@H1][Branch1][C][C][NH2+1][C][=Branch1][C][=O][C][=C][C][=C][C][=C][Ring1][=Branch1][F]
优化后 SMILES: Cc1nn(C)c(C)c1[C@@H](C)[NH2+]C(=O)c1ccccc1F
相似度: 0.2777777777777778
QED: 0.9262729281846569 dQED: 0.1919763972132874  srcQED:0.73430

`main/train.py`训练日志：

有效分子数: 249455
构造高相似训练对...
训练对数量: 15560
Pair quality | AvgSim 0.397 | AvgDQED +0.117 | MinSim 0.350 | MinDQED +0.060
词表大小: 59
<div style="overflow-x: auto;">
  <table style="border-collapse: collapse; width: 100%; font-size: 13px;">
    <thead>
      <tr style="background-color: #f2f2f2;">
        <th style="border: 1px solid #ddd; padding: 6px;">Epoch</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Train Loss</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Val Loss</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Validity</th>
        <th style="border: 1px solid #ddd; padding: 6px;">SimSrc</th>
        <th style="border: 1px solid #ddd; padding: 6px;">SimTgt</th>
        <th style="border: 1px solid #ddd; padding: 6px;">dQED</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Opt30</th>
        <th style="border: 1px solid #ddd; padding: 6px;">QEDLoss</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Score</th>
      </tr>
    </thead>
    <tbody>
      <tr><td style="border:1px solid #ddd; padding:4px;">001</td><td style="border:1px solid #ddd; padding:4px;">1.8039</td><td style="border:1px solid #ddd; padding:4px;">1.1862</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.147</td><td style="border:1px solid #ddd; padding:4px;">0.139</td><td style="border:1px solid #ddd; padding:4px;">-0.356</td><td style="border:1px solid #ddd; padding:4px;">0.008</td><td style="border:1px solid #ddd; padding:4px;">0.0067</td><td style="border:1px solid #ddd; padding:4px;">-0.530</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">002</td><td style="border:1px solid #ddd; padding:4px;">1.4830</td><td style="border:1px solid #ddd; padding:4px;">1.0423</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.142</td><td style="border:1px solid #ddd; padding:4px;">0.134</td><td style="border:1px solid #ddd; padding:4px;">-0.364</td><td style="border:1px solid #ddd; padding:4px;">0.012</td><td style="border:1px solid #ddd; padding:4px;">0.0036</td><td style="border:1px solid #ddd; padding:4px;">-0.577</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">003</td><td style="border:1px solid #ddd; padding:4px;">1.3877</td><td style="border:1px solid #ddd; padding:4px;">0.9651</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.154</td><td style="border:1px solid #ddd; padding:4px;">0.143</td><td style="border:1px solid #ddd; padding:4px;">-0.315</td><td style="border:1px solid #ddd; padding:4px;">0.026</td><td style="border:1px solid #ddd; padding:4px;">0.0033</td><td style="border:1px solid #ddd; padding:4px;">-0.035</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">004</td><td style="border:1px solid #ddd; padding:4px;">1.3324</td><td style="border:1px solid #ddd; padding:4px;">0.9298</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.166</td><td style="border:1px solid #ddd; padding:4px;">0.150</td><td style="border:1px solid #ddd; padding:4px;">-0.348</td><td style="border:1px solid #ddd; padding:4px;">0.048</td><td style="border:1px solid #ddd; padding:4px;">0.0030</td><td style="border:1px solid #ddd; padding:4px;">-0.250</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">005</td><td style="border:1px solid #ddd; padding:4px;">1.2644</td><td style="border:1px solid #ddd; padding:4px;">0.8816</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.196</td><td style="border:1px solid #ddd; padding:4px;">0.171</td><td style="border:1px solid #ddd; padding:4px;">-0.218</td><td style="border:1px solid #ddd; padding:4px;">0.091</td><td style="border:1px solid #ddd; padding:4px;">0.0030</td><td style="border:1px solid #ddd; padding:4px;">1.019</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">006</td><td style="border:1px solid #ddd; padding:4px;">1.2105</td><td style="border:1px solid #ddd; padding:4px;">0.8599</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.197</td><td style="border:1px solid #ddd; padding:4px;">0.172</td><td style="border:1px solid #ddd; padding:4px;">-0.115</td><td style="border:1px solid #ddd; padding:4px;">0.109</td><td style="border:1px solid #ddd; padding:4px;">0.0027</td><td style="border:1px solid #ddd; padding:4px;">1.974</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">007</td><td style="border:1px solid #ddd; padding:4px;">1.1820</td><td style="border:1px solid #ddd; padding:4px;">0.8283</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.221</td><td style="border:1px solid #ddd; padding:4px;">0.187</td><td style="border:1px solid #ddd; padding:4px;">-0.165</td><td style="border:1px solid #ddd; padding:4px;">0.136</td><td style="border:1px solid #ddd; padding:4px;">0.0026</td><td style="border:1px solid #ddd; padding:4px;">1.673</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">008</td><td style="border:1px solid #ddd; padding:4px;">1.1615</td><td style="border:1px solid #ddd; padding:4px;">0.8276</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.228</td><td style="border:1px solid #ddd; padding:4px;">0.187</td><td style="border:1px solid #ddd; padding:4px;">-0.164</td><td style="border:1px solid #ddd; padding:4px;">0.147</td><td style="border:1px solid #ddd; padding:4px;">0.0025</td><td style="border:1px solid #ddd; padding:4px;">1.703</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">009</td><td style="border:1px solid #ddd; padding:4px;">1.1205</td><td style="border:1px solid #ddd; padding:4px;">0.8169</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.197</td><td style="border:1px solid #ddd; padding:4px;">0.162</td><td style="border:1px solid #ddd; padding:4px;">-0.344</td><td style="border:1px solid #ddd; padding:4px;">0.092</td><td style="border:1px solid #ddd; padding:4px;">0.0023</td><td style="border:1px solid #ddd; padding:4px;">-0.030</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">010</td><td style="border:1px solid #ddd; padding:4px;">1.0920</td><td style="border:1px solid #ddd; padding:4px;">0.7915</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.236</td><td style="border:1px solid #ddd; padding:4px;">0.196</td><td style="border:1px solid #ddd; padding:4px;">-0.127</td><td style="border:1px solid #ddd; padding:4px;">0.162</td><td style="border:1px solid #ddd; padding:4px;">0.0023</td><td style="border:1px solid #ddd; padding:4px;">2.089</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">011</td><td style="border:1px solid #ddd; padding:4px;">1.0728</td><td style="border:1px solid #ddd; padding:4px;">0.7982</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.233</td><td style="border:1px solid #ddd; padding:4px;">0.187</td><td style="border:1px solid #ddd; padding:4px;">-0.131</td><td style="border:1px solid #ddd; padding:4px;">0.161</td><td style="border:1px solid #ddd; padding:4px;">0.0022</td><td style="border:1px solid #ddd; padding:4px;">2.050</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">012</td><td style="border:1px solid #ddd; padding:4px;">1.0570</td><td style="border:1px solid #ddd; padding:4px;">0.7801</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.228</td><td style="border:1px solid #ddd; padding:4px;">0.188</td><td style="border:1px solid #ddd; padding:4px;">-0.138</td><td style="border:1px solid #ddd; padding:4px;">0.154</td><td style="border:1px solid #ddd; padding:4px;">0.0021</td><td style="border:1px solid #ddd; padding:4px;">1.953</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">013</td><td style="border:1px solid #ddd; padding:4px;">1.0187</td><td style="border:1px solid #ddd; padding:4px;">0.7842</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.265</td><td style="border:1px solid #ddd; padding:4px;">0.210</td><td style="border:1px solid #ddd; padding:4px;">-0.104</td><td style="border:1px solid #ddd; padding:4px;">0.231</td><td style="border:1px solid #ddd; padding:4px;">0.0020</td><td style="border:1px solid #ddd; padding:4px;">2.497</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">014</td><td style="border:1px solid #ddd; padding:4px;">0.9938</td><td style="border:1px solid #ddd; padding:4px;">0.7740</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.258</td><td style="border:1px solid #ddd; padding:4px;">0.208</td><td style="border:1px solid #ddd; padding:4px;">-0.056</td><td style="border:1px solid #ddd; padding:4px;">0.214</td><td style="border:1px solid #ddd; padding:4px;">0.0019</td><td style="border:1px solid #ddd; padding:4px;">2.884</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">015</td><td style="border:1px solid #ddd; padding:4px;">0.9833</td><td style="border:1px solid #ddd; padding:4px;">0.7829</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.246</td><td style="border:1px solid #ddd; padding:4px;">0.205</td><td style="border:1px solid #ddd; padding:4px;">-0.111</td><td style="border:1px solid #ddd; padding:4px;">0.196</td><td style="border:1px solid #ddd; padding:4px;">0.0019</td><td style="border:1px solid #ddd; padding:4px;">2.333</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">016</td><td style="border:1px solid #ddd; padding:4px;">0.9565</td><td style="border:1px solid #ddd; padding:4px;">0.7744</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.263</td><td style="border:1px solid #ddd; padding:4px;">0.216</td><td style="border:1px solid #ddd; padding:4px;">-0.018</td><td style="border:1px solid #ddd; padding:4px;">0.232</td><td style="border:1px solid #ddd; padding:4px;">0.0019</td><td style="border:1px solid #ddd; padding:4px;">3.259</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">017</td><td style="border:1px solid #ddd; padding:4px;">0.9242</td><td style="border:1px solid #ddd; padding:4px;">0.7757</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.252</td><td style="border:1px solid #ddd; padding:4px;">0.203</td><td style="border:1px solid #ddd; padding:4px;">-0.081</td><td style="border:1px solid #ddd; padding:4px;">0.202</td><td style="border:1px solid #ddd; padding:4px;">0.0018</td><td style="border:1px solid #ddd; padding:4px;">2.615</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">018</td><td style="border:1px solid #ddd; padding:4px;">0.9310</td><td style="border:1px solid #ddd; padding:4px;">0.7731</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.254</td><td style="border:1px solid #ddd; padding:4px;">0.212</td><td style="border:1px solid #ddd; padding:4px;">-0.012</td><td style="border:1px solid #ddd; padding:4px;">0.229</td><td style="border:1px solid #ddd; padding:4px;">0.0017</td><td style="border:1px solid #ddd; padding:4px;">3.298</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">019</td><td style="border:1px solid #ddd; padding:4px;">0.8959</td><td style="border:1px solid #ddd; padding:4px;">0.7803</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.260</td><td style="border:1px solid #ddd; padding:4px;">0.213</td><td style="border:1px solid #ddd; padding:4px;">-0.044</td><td style="border:1px solid #ddd; padding:4px;">0.237</td><td style="border:1px solid #ddd; padding:4px;">0.0016</td><td style="border:1px solid #ddd; padding:4px;">3.051</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">020</td><td style="border:1px solid #ddd; padding:4px;">0.8799</td><td style="border:1px solid #ddd; padding:4px;">0.7912</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.248</td><td style="border:1px solid #ddd; padding:4px;">0.206</td><td style="border:1px solid #ddd; padding:4px;">-0.041</td><td style="border:1px solid #ddd; padding:4px;">0.206</td><td style="border:1px solid #ddd; padding:4px;">0.0016</td><td style="border:1px solid #ddd; padding:4px;">2.964</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">021</td><td style="border:1px solid #ddd; padding:4px;">0.8528</td><td style="border:1px solid #ddd; padding:4px;">0.8007</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.257</td><td style="border:1px solid #ddd; padding:4px;">0.210</td><td style="border:1px solid #ddd; padding:4px;">-0.020</td><td style="border:1px solid #ddd; padding:4px;">0.229</td><td style="border:1px solid #ddd; padding:4px;">0.0016</td><td style="border:1px solid #ddd; padding:4px;">3.211</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">022</td><td style="border:1px solid #ddd; padding:4px;">0.8434</td><td style="border:1px solid #ddd; padding:4px;">0.7913</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.262</td><td style="border:1px solid #ddd; padding:4px;">0.214</td><td style="border:1px solid #ddd; padding:4px;">-0.027</td><td style="border:1px solid #ddd; padding:4px;">0.235</td><td style="border:1px solid #ddd; padding:4px;">0.0015</td><td style="border:1px solid #ddd; padding:4px;">3.188</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">023</td><td style="border:1px solid #ddd; padding:4px;">0.8219</td><td style="border:1px solid #ddd; padding:4px;">0.8076</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.259</td><td style="border:1px solid #ddd; padding:4px;">0.213</td><td style="border:1px solid #ddd; padding:4px;">-0.053</td><td style="border:1px solid #ddd; padding:4px;">0.238</td><td style="border:1px solid #ddd; padding:4px;">0.0016</td><td style="border:1px solid #ddd; padding:4px;">2.955</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">024</td><td style="border:1px solid #ddd; padding:4px;">0.7822</td><td style="border:1px solid #ddd; padding:4px;">0.8016</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.265</td><td style="border:1px solid #ddd; padding:4px;">0.219</td><td style="border:1px solid #ddd; padding:4px;">-0.006</td><td style="border:1px solid #ddd; padding:4px;">0.253</td><td style="border:1px solid #ddd; padding:4px;">0.0015</td><td style="border:1px solid #ddd; padding:4px;">3.403</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">025</td><td style="border:1px solid #ddd; padding:4px;">0.7671</td><td style="border:1px solid #ddd; padding:4px;">0.8129</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.246</td><td style="border:1px solid #ddd; padding:4px;">0.207</td><td style="border:1px solid #ddd; padding:4px;">-0.059</td><td style="border:1px solid #ddd; padding:4px;">0.225</td><td style="border:1px solid #ddd; padding:4px;">0.0015</td><td style="border:1px solid #ddd; padding:4px;">2.836</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">026</td><td style="border:1px solid #ddd; padding:4px;">0.7383</td><td style="border:1px solid #ddd; padding:4px;">0.8150</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.277</td><td style="border:1px solid #ddd; padding:4px;">0.230</td><td style="border:1px solid #ddd; padding:4px;">0.011</td><td style="border:1px solid #ddd; padding:4px;">0.317</td><td style="border:1px solid #ddd; padding:4px;">0.0014</td><td style="border:1px solid #ddd; padding:4px;">3.714</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">027</td><td style="border:1px solid #ddd; padding:4px;">0.7318</td><td style="border:1px solid #ddd; padding:4px;">0.8279</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.260</td><td style="border:1px solid #ddd; padding:4px;">0.216</td><td style="border:1px solid #ddd; padding:4px;">0.009</td><td style="border:1px solid #ddd; padding:4px;">0.278</td><td style="border:1px solid #ddd; padding:4px;">0.0014</td><td style="border:1px solid #ddd; padding:4px;">3.568</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">028</td><td style="border:1px solid #ddd; padding:4px;">0.7002</td><td style="border:1px solid #ddd; padding:4px;">0.8354</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.262</td><td style="border:1px solid #ddd; padding:4px;">0.220</td><td style="border:1px solid #ddd; padding:4px;">0.009</td><td style="border:1px solid #ddd; padding:4px;">0.285</td><td style="border:1px solid #ddd; padding:4px;">0.0014</td><td style="border:1px solid #ddd; padding:4px;">3.585</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">029</td><td style="border:1px solid #ddd; padding:4px;">0.6874</td><td style="border:1px solid #ddd; padding:4px;">0.8423</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.262</td><td style="border:1px solid #ddd; padding:4px;">0.218</td><td style="border:1px solid #ddd; padding:4px;">0.006</td><td style="border:1px solid #ddd; padding:4px;">0.287</td><td style="border:1px solid #ddd; padding:4px;">0.0013</td><td style="border:1px solid #ddd; padding:4px;">3.583</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">030</td><td style="border:1px solid #ddd; padding:4px;">0.6639</td><td style="border:1px solid #ddd; padding:4px;">0.8514</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.261</td><td style="border:1px solid #ddd; padding:4px;">0.218</td><td style="border:1px solid #ddd; padding:4px;">0.016</td><td style="border:1px solid #ddd; padding:4px;">0.282</td><td style="border:1px solid #ddd; padding:4px;">0.0013</td><td style="border:1px solid #ddd; padding:4px;">3.640</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">031</td><td style="border:1px solid #ddd; padding:4px;">0.6365</td><td style="border:1px solid #ddd; padding:4px;">0.8620</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.273</td><td style="border:1px solid #ddd; padding:4px;">0.224</td><td style="border:1px solid #ddd; padding:4px;">0.013</td><td style="border:1px solid #ddd; padding:4px;">0.314</td><td style="border:1px solid #ddd; padding:4px;">0.0013</td><td style="border:1px solid #ddd; padding:4px;">3.707</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">032</td><td style="border:1px solid #ddd; padding:4px;">0.6160</td><td style="border:1px solid #ddd; padding:4px;">0.8663</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.274</td><td style="border:1px solid #ddd; padding:4px;">0.228</td><td style="border:1px solid #ddd; padding:4px;">0.020</td><td style="border:1px solid #ddd; padding:4px;">0.326</td><td style="border:1px solid #ddd; padding:4px;">0.0013</td><td style="border:1px solid #ddd; padding:4px;">3.806</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">033</td><td style="border:1px solid #ddd; padding:4px;">0.5901</td><td style="border:1px solid #ddd; padding:4px;">0.8746</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.275</td><td style="border:1px solid #ddd; padding:4px;">0.229</td><td style="border:1px solid #ddd; padding:4px;">0.019</td><td style="border:1px solid #ddd; padding:4px;">0.335</td><td style="border:1px solid #ddd; padding:4px;">0.0013</td><td style="border:1px solid #ddd; padding:4px;">3.799</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">034</td><td style="border:1px solid #ddd; padding:4px;">0.5928</td><td style="border:1px solid #ddd; padding:4px;">0.8977</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.266</td><td style="border:1px solid #ddd; padding:4px;">0.223</td><td style="border:1px solid #ddd; padding:4px;">0.016</td><td style="border:1px solid #ddd; padding:4px;">0.344</td><td style="border:1px solid #ddd; padding:4px;">0.0012</td><td style="border:1px solid #ddd; padding:4px;">3.760</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">035</td><td style="border:1px solid #ddd; padding:4px;">0.5873</td><td style="border:1px solid #ddd; padding:4px;">0.9151</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.275</td><td style="border:1px solid #ddd; padding:4px;">0.227</td><td style="border:1px solid #ddd; padding:4px;">0.003</td><td style="border:1px solid #ddd; padding:4px;">0.344</td><td style="border:1px solid #ddd; padding:4px;">0.0012</td><td style="border:1px solid #ddd; padding:4px;">3.649</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">036</td><td style="border:1px solid #ddd; padding:4px;">0.5732</td><td style="border:1px solid #ddd; padding:4px;">0.9037</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.277</td><td style="border:1px solid #ddd; padding:4px;">0.230</td><td style="border:1px solid #ddd; padding:4px;">0.026</td><td style="border:1px solid #ddd; padding:4px;">0.358</td><td style="border:1px solid #ddd; padding:4px;">0.0012</td><td style="border:1px solid #ddd; padding:4px;">3.906</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">037</td><td style="border:1px solid #ddd; padding:4px;">0.5499</td><td style="border:1px solid #ddd; padding:4px;">0.9118</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.280</td><td style="border:1px solid #ddd; padding:4px;">0.231</td><td style="border:1px solid #ddd; padding:4px;">0.007</td><td style="border:1px solid #ddd; padding:4px;">0.364</td><td style="border:1px solid #ddd; padding:4px;">0.0012</td><td style="border:1px solid #ddd; padding:4px;">3.730</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">038</td><td style="border:1px solid #ddd; padding:4px;">0.5279</td><td style="border:1px solid #ddd; padding:4px;">0.9322</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.286</td><td style="border:1px solid #ddd; padding:4px;">0.235</td><td style="border:1px solid #ddd; padding:4px;">0.027</td><td style="border:1px solid #ddd; padding:4px;">0.392</td><td style="border:1px solid #ddd; padding:4px;">0.0012</td><td style="border:1px solid #ddd; padding:4px;">4.000</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">039</td><td style="border:1px solid #ddd; padding:4px;">0.5223</td><td style="border:1px solid #ddd; padding:4px;">0.9331</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.286</td><td style="border:1px solid #ddd; padding:4px;">0.236</td><td style="border:1px solid #ddd; padding:4px;">0.025</td><td style="border:1px solid #ddd; padding:4px;">0.399</td><td style="border:1px solid #ddd; padding:4px;">0.0012</td><td style="border:1px solid #ddd; padding:4px;">3.975</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">040</td><td style="border:1px solid #ddd; padding:4px;">0.4319</td><td style="border:1px solid #ddd; padding:4px;">0.9427</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.296</td><td style="border:1px solid #ddd; padding:4px;">0.243</td><td style="border:1px solid #ddd; padding:4px;">0.040</td><td style="border:1px solid #ddd; padding:4px;">0.487</td><td style="border:1px solid #ddd; padding:4px;">0.0010</td><td style="border:1px solid #ddd; padding:4px;">4.292</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">041</td><td style="border:1px solid #ddd; padding:4px;">0.3844</td><td style="border:1px solid #ddd; padding:4px;">0.9592</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.306</td><td style="border:1px solid #ddd; padding:4px;">0.249</td><td style="border:1px solid #ddd; padding:4px;">0.052</td><td style="border:1px solid #ddd; padding:4px;">0.506</td><td style="border:1px solid #ddd; padding:4px;">0.0010</td><td style="border:1px solid #ddd; padding:4px;">4.475</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">042</td><td style="border:1px solid #ddd; padding:4px;">0.3730</td><td style="border:1px solid #ddd; padding:4px;">0.9746</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.310</td><td style="border:1px solid #ddd; padding:4px;">0.252</td><td style="border:1px solid #ddd; padding:4px;">0.056</td><td style="border:1px solid #ddd; padding:4px;">0.531</td><td style="border:1px solid #ddd; padding:4px;">0.0009</td><td style="border:1px solid #ddd; padding:4px;">4.547</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">043</td><td style="border:1px solid #ddd; padding:4px;">0.3600</td><td style="border:1px solid #ddd; padding:4px;">0.9853</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.310</td><td style="border:1px solid #ddd; padding:4px;">0.251</td><td style="border:1px solid #ddd; padding:4px;">0.056</td><td style="border:1px solid #ddd; padding:4px;">0.531</td><td style="border:1px solid #ddd; padding:4px;">0.0009</td><td style="border:1px solid #ddd; padding:4px;">4.546</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">044</td><td style="border:1px solid #ddd; padding:4px;">0.3423</td><td style="border:1px solid #ddd; padding:4px;">1.0000</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.312</td><td style="border:1px solid #ddd; padding:4px;">0.255</td><td style="border:1px solid #ddd; padding:4px;">0.055</td><td style="border:1px solid #ddd; padding:4px;">0.548</td><td style="border:1px solid #ddd; padding:4px;">0.0009</td><td style="border:1px solid #ddd; padding:4px;">4.578</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">045</td><td style="border:1px solid #ddd; padding:4px;">0.3400</td><td style="border:1px solid #ddd; padding:4px;">1.0077</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.311</td><td style="border:1px solid #ddd; padding:4px;">0.253</td><td style="border:1px solid #ddd; padding:4px;">0.053</td><td style="border:1px solid #ddd; padding:4px;">0.563</td><td style="border:1px solid #ddd; padding:4px;">0.0009</td><td style="border:1px solid #ddd; padding:4px;">4.565</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">046</td><td style="border:1px solid #ddd; padding:4px;">0.3269</td><td style="border:1px solid #ddd; padding:4px;">1.0221</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.318</td><td style="border:1px solid #ddd; padding:4px;">0.259</td><td style="border:1px solid #ddd; padding:4px;">0.058</td><td style="border:1px solid #ddd; padding:4px;">0.576</td><td style="border:1px solid #ddd; padding:4px;">0.0009</td><td style="border:1px solid #ddd; padding:4px;">4.658</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">047</td><td style="border:1px solid #ddd; padding:4px;">0.3093</td><td style="border:1px solid #ddd; padding:4px;">1.0232</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.315</td><td style="border:1px solid #ddd; padding:4px;">0.255</td><td style="border:1px solid #ddd; padding:4px;">0.059</td><td style="border:1px solid #ddd; padding:4px;">0.575</td><td style="border:1px solid #ddd; padding:4px;">0.0009</td><td style="border:1px solid #ddd; padding:4px;">4.646</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">048</td><td style="border:1px solid #ddd; padding:4px;">0.3169</td><td style="border:1px solid #ddd; padding:4px;">1.0483</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.315</td><td style="border:1px solid #ddd; padding:4px;">0.255</td><td style="border:1px solid #ddd; padding:4px;">0.054</td><td style="border:1px solid #ddd; padding:4px;">0.573</td><td style="border:1px solid #ddd; padding:4px;">0.0009</td><td style="border:1px solid #ddd; padding:4px;">4.590</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">049</td><td style="border:1px solid #ddd; padding:4px;">0.3058</td><td style="border:1px solid #ddd; padding:4px;">1.0420</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.317</td><td style="border:1px solid #ddd; padding:4px;">0.259</td><td style="border:1px solid #ddd; padding:4px;">0.063</td><td style="border:1px solid #ddd; padding:4px;">0.598</td><td style="border:1px solid #ddd; padding:4px;">0.0008</td><td style="border:1px solid #ddd; padding:4px;">4.731</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">050</td><td style="border:1px solid #ddd; padding:4px;">0.2881</td><td style="border:1px solid #ddd; padding:4px;">1.0534</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.320</td><td style="border:1px solid #ddd; padding:4px;">0.259</td><td style="border:1px solid #ddd; padding:4px;">0.060</td><td style="border:1px solid #ddd; padding:4px;">0.603</td><td style="border:1px solid #ddd; padding:4px;">0.0008</td><td style="border:1px solid #ddd; padding:4px;">4.714</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">051</td><td style="border:1px solid #ddd; padding:4px;">0.2958</td><td style="border:1px solid #ddd; padding:4px;">1.0647</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.316</td><td style="border:1px solid #ddd; padding:4px;">0.256</td><td style="border:1px solid #ddd; padding:4px;">0.054</td><td style="border:1px solid #ddd; padding:4px;">0.602</td><td style="border:1px solid #ddd; padding:4px;">0.0008</td><td style="border:1px solid #ddd; padding:4px;">4.637</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">052</td><td style="border:1px solid #ddd; padding:4px;">0.2969</td><td style="border:1px solid #ddd; padding:4px;">1.0591</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.315</td><td style="border:1px solid #ddd; padding:4px;">0.255</td><td style="border:1px solid #ddd; padding:4px;">0.065</td><td style="border:1px solid #ddd; padding:4px;">0.591</td><td style="border:1px solid #ddd; padding:4px;">0.0008</td><td style="border:1px solid #ddd; padding:4px;">4.734</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">053</td><td style="border:1px solid #ddd; padding:4px;">0.2859</td><td style="border:1px solid #ddd; padding:4px;">1.0745</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.323</td><td style="border:1px solid #ddd; padding:4px;">0.260</td><td style="border:1px solid #ddd; padding:4px;">0.058</td><td style="border:1px solid #ddd; padding:4px;">0.606</td><td style="border:1px solid #ddd; padding:4px;">0.0008</td><td style="border:1px solid #ddd; padding:4px;">4.692</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">054</td><td style="border:1px solid #ddd; padding:4px;">0.2851</td><td style="border:1px solid #ddd; padding:4px;">1.0801</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.317</td><td style="border:1px solid #ddd; padding:4px;">0.258</td><td style="border:1px solid #ddd; padding:4px;">0.056</td><td style="border:1px solid #ddd; padding:4px;">0.603</td><td style="border:1px solid #ddd; padding:4px;">0.0008</td><td style="border:1px solid #ddd; padding:4px;">4.661</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">055</td><td style="border:1px solid #ddd; padding:4px;">0.2766</td><td style="border:1px solid #ddd; padding:4px;">1.0850</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.315</td><td style="border:1px solid #ddd; padding:4px;">0.259</td><td style="border:1px solid #ddd; padding:4px;">0.062</td><td style="border:1px solid #ddd; padding:4px;">0.596</td><td style="border:1px solid #ddd; padding:4px;">0.0008</td><td style="border:1px solid #ddd; padding:4px;">4.693</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">056</td><td style="border:1px solid #ddd; padding:4px;">0.2666</td><td style="border:1px solid #ddd; padding:4px;">1.0964</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.319</td><td style="border:1px solid #ddd; padding:4px;">0.259</td><td style="border:1px solid #ddd; padding:4px;">0.061</td><td style="border:1px solid #ddd; padding:4px;">0.608</td><td style="border:1px solid #ddd; padding:4px;">0.0008</td><td style="border:1px solid #ddd; padding:4px;">4.709</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">057</td><td style="border:1px solid #ddd; padding:4px;">0.2633</td><td style="border:1px solid #ddd; padding:4px;">1.1068</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.318</td><td style="border:1px solid #ddd; padding:4px;">0.258</td><td style="border:1px solid #ddd; padding:4px;">0.058</td><td style="border:1px solid #ddd; padding:4px;">0.607</td><td style="border:1px solid #ddd; padding:4px;">0.0008</td><td style="border:1px solid #ddd; padding:4px;">4.691</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">058</td><td style="border:1px solid #ddd; padding:4px;">0.2673</td><td style="border:1px solid #ddd; padding:4px;">1.0971</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.317</td><td style="border:1px solid #ddd; padding:4px;">0.257</td><td style="border:1px solid #ddd; padding:4px;">0.062</td><td style="border:1px solid #ddd; padding:4px;">0.605</td><td style="border:1px solid #ddd; padding:4px;">0.0008</td><td style="border:1px solid #ddd; padding:4px;">4.712</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">059</td><td style="border:1px solid #ddd; padding:4px;">0.2686</td><td style="border:1px solid #ddd; padding:4px;">1.1072</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.319</td><td style="border:1px solid #ddd; padding:4px;">0.257</td><td style="border:1px solid #ddd; padding:4px;">0.059</td><td style="border:1px solid #ddd; padding:4px;">0.609</td><td style="border:1px solid #ddd; padding:4px;">0.0008</td><td style="border:1px solid #ddd; padding:4px;">4.693</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">060</td><td style="border:1px solid #ddd; padding:4px;">0.2500</td><td style="border:1px solid #ddd; padding:4px;">1.1231</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.324</td><td style="border:1px solid #ddd; padding:4px;">0.261</td><td style="border:1px solid #ddd; padding:4px;">0.064</td><td style="border:1px solid #ddd; padding:4px;">0.636</td><td style="border:1px solid #ddd; padding:4px;">0.0007</td><td style="border:1px solid #ddd; padding:4px;">4.781</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">061</td><td style="border:1px solid #ddd; padding:4px;">0.2142</td><td style="border:1px solid #ddd; padding:4px;">1.1223</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.328</td><td style="border:1px solid #ddd; padding:4px;">0.264</td><td style="border:1px solid #ddd; padding:4px;">0.066</td><td style="border:1px solid #ddd; padding:4px;">0.652</td><td style="border:1px solid #ddd; padding:4px;">0.0007</td><td style="border:1px solid #ddd; padding:4px;">4.862</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">062</td><td style="border:1px solid #ddd; padding:4px;">0.1958</td><td style="border:1px solid #ddd; padding:4px;">1.1284</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.327</td><td style="border:1px solid #ddd; padding:4px;">0.265</td><td style="border:1px solid #ddd; padding:4px;">0.068</td><td style="border:1px solid #ddd; padding:4px;">0.643</td><td style="border:1px solid #ddd; padding:4px;">0.0007</td><td style="border:1px solid #ddd; padding:4px;">4.851</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">063</td><td style="border:1px solid #ddd; padding:4px;">0.1896</td><td style="border:1px solid #ddd; padding:4px;">1.1410</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.328</td><td style="border:1px solid #ddd; padding:4px;">0.266</td><td style="border:1px solid #ddd; padding:4px;">0.068</td><td style="border:1px solid #ddd; padding:4px;">0.647</td><td style="border:1px solid #ddd; padding:4px;">0.0006</td><td style="border:1px solid #ddd; padding:4px;">4.868</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">064</td><td style="border:1px solid #ddd; padding:4px;">0.1902</td><td style="border:1px solid #ddd; padding:4px;">1.1529</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.330</td><td style="border:1px solid #ddd; padding:4px;">0.265</td><td style="border:1px solid #ddd; padding:4px;">0.064</td><td style="border:1px solid #ddd; padding:4px;">0.643</td><td style="border:1px solid #ddd; padding:4px;">0.0006</td><td style="border:1px solid #ddd; padding:4px;">4.823</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">065</td><td style="border:1px solid #ddd; padding:4px;">0.1884</td><td style="border:1px solid #ddd; padding:4px;">1.1594</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.328</td><td style="border:1px solid #ddd; padding:4px;">0.264</td><td style="border:1px solid #ddd; padding:4px;">0.069</td><td style="border:1px solid #ddd; padding:4px;">0.650</td><td style="border:1px solid #ddd; padding:4px;">0.0006</td><td style="border:1px solid #ddd; padding:4px;">4.872</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">066</td><td style="border:1px solid #ddd; padding:4px;">0.1804</td><td style="border:1px solid #ddd; padding:4px;">1.1642</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.329</td><td style="border:1px solid #ddd; padding:4px;">0.267</td><td style="border:1px solid #ddd; padding:4px;">0.070</td><td style="border:1px solid #ddd; padding:4px;">0.654</td><td style="border:1px solid #ddd; padding:4px;">0.0006</td><td style="border:1px solid #ddd; padding:4px;">4.896</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">067</td><td style="border:1px solid #ddd; padding:4px;">0.1888</td><td style="border:1px solid #ddd; padding:4px;">1.1619</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.329</td><td style="border:1px solid #ddd; padding:4px;">0.266</td><td style="border:1px solid #ddd; padding:4px;">0.072</td><td style="border:1px solid #ddd; padding:4px;">0.663</td><td style="border:1px solid #ddd; padding:4px;">0.0006</td><td style="border:1px solid #ddd; padding:4px;">4.934</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">068</td><td style="border:1px solid #ddd; padding:4px;">0.1781</td><td style="border:1px solid #ddd; padding:4px;">1.1714</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.331</td><td style="border:1px solid #ddd; padding:4px;">0.266</td><td style="border:1px solid #ddd; padding:4px;">0.073</td><td style="border:1px solid #ddd; padding:4px;">0.661</td><td style="border:1px solid #ddd; padding:4px;">0.0006</td><td style="border:1px solid #ddd; padding:4px;">4.939</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">069</td><td style="border:1px solid #ddd; padding:4px;">0.1858</td><td style="border:1px solid #ddd; padding:4px;">1.1744</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.328</td><td style="border:1px solid #ddd; padding:4px;">0.265</td><td style="border:1px solid #ddd; padding:4px;">0.072</td><td style="border:1px solid #ddd; padding:4px;">0.656</td><td style="border:1px solid #ddd; padding:4px;">0.0006</td><td style="border:1px solid #ddd; padding:4px;">4.909</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">070</td><td style="border:1px solid #ddd; padding:4px;">0.1725</td><td style="border:1px solid #ddd; padding:4px;">1.1793</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.327</td><td style="border:1px solid #ddd; padding:4px;">0.265</td><td style="border:1px solid #ddd; padding:4px;">0.073</td><td style="border:1px solid #ddd; padding:4px;">0.652</td><td style="border:1px solid #ddd; padding:4px;">0.0006</td><td style="border:1px solid #ddd; padding:4px;">4.905</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">071</td><td style="border:1px solid #ddd; padding:4px;">0.1886</td><td style="border:1px solid #ddd; padding:4px;">1.1809</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.329</td><td style="border:1px solid #ddd; padding:4px;">0.265</td><td style="border:1px solid #ddd; padding:4px;">0.070</td><td style="border:1px solid #ddd; padding:4px;">0.649</td><td style="border:1px solid #ddd; padding:4px;">0.0006</td><td style="border:1px solid #ddd; padding:4px;">4.877</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">072</td><td style="border:1px solid #ddd; padding:4px;">0.1852</td><td style="border:1px solid #ddd; padding:4px;">1.1806</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.326</td><td style="border:1px solid #ddd; padding:4px;">0.263</td><td style="border:1px solid #ddd; padding:4px;">0.070</td><td style="border:1px solid #ddd; padding:4px;">0.649</td><td style="border:1px solid #ddd; padding:4px;">0.0006</td><td style="border:1px solid #ddd; padding:4px;">4.870</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">073</td><td style="border:1px solid #ddd; padding:4px;">0.1785</td><td style="border:1px solid #ddd; padding:4px;">1.1848</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.327</td><td style="border:1px solid #ddd; padding:4px;">0.263</td><td style="border:1px solid #ddd; padding:4px;">0.069</td><td style="border:1px solid #ddd; padding:4px;">0.660</td><td style="border:1px solid #ddd; padding:4px;">0.0006</td><td style="border:1px solid #ddd; padding:4px;">4.881</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">074</td><td style="border:1px solid #ddd; padding:4px;">0.1751</td><td style="border:1px solid #ddd; padding:4px;">1.1923</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.331</td><td style="border:1px solid #ddd; padding:4px;">0.266</td><td style="border:1px solid #ddd; padding:4px;">0.069</td><td style="border:1px solid #ddd; padding:4px;">0.656</td><td style="border:1px solid #ddd; padding:4px;">0.0006</td><td style="border:1px solid #ddd; padding:4px;">4.884</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">075</td><td style="border:1px solid #ddd; padding:4px;">0.1760</td><td style="border:1px solid #ddd; padding:4px;">1.1984</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.326</td><td style="border:1px solid #ddd; padding:4px;">0.263</td><td style="border:1px solid #ddd; padding:4px;">0.071</td><td style="border:1px solid #ddd; padding:4px;">0.654</td><td style="border:1px solid #ddd; padding:4px;">0.0006</td><td style="border:1px solid #ddd; padding:4px;">4.888</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">076</td><td style="border:1px solid #ddd; padding:4px;">0.1737</td><td style="border:1px solid #ddd; padding:4px;">1.1969</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.327</td><td style="border:1px solid #ddd; padding:4px;">0.263</td><td style="border:1px solid #ddd; padding:4px;">0.072</td><td style="border:1px solid #ddd; padding:4px;">0.654</td><td style="border:1px solid #ddd; padding:4px;">0.0006</td><td style="border:1px solid #ddd; padding:4px;">4.887</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">077</td><td style="border:1px solid #ddd; padding:4px;">0.1747</td><td style="border:1px solid #ddd; padding:4px;">1.2005</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.328</td><td style="border:1px solid #ddd; padding:4px;">0.264</td><td style="border:1px solid #ddd; padding:4px;">0.063</td><td style="border:1px solid #ddd; padding:4px;">0.653</td><td style="border:1px solid #ddd; padding:4px;">0.0006</td><td style="border:1px solid #ddd; padding:4px;">4.817</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">078</td><td style="border:1px solid #ddd; padding:4px;">0.1740</td><td style="border:1px solid #ddd; padding:4px;">1.1993</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.327</td><td style="border:1px solid #ddd; padding:4px;">0.264</td><td style="border:1px solid #ddd; padding:4px;">0.068</td><td style="border:1px solid #ddd; padding:4px;">0.648</td><td style="border:1px solid #ddd; padding:4px;">0.0006</td><td style="border:1px solid #ddd; padding:4px;">4.855</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">079</td><td style="border:1px solid #ddd; padding:4px;">0.1654</td><td style="border:1px solid #ddd; padding:4px;">1.2141</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.327</td><td style="border:1px solid #ddd; padding:4px;">0.263</td><td style="border:1px solid #ddd; padding:4px;">0.073</td><td style="border:1px solid #ddd; padding:4px;">0.656</td><td style="border:1px solid #ddd; padding:4px;">0.0005</td><td style="border:1px solid #ddd; padding:4px;">4.905</td></tr>
      <tr><td style="border:1px solid #ddd; padding:4px;">080</td><td style="border:1px solid #ddd; padding:4px;">0.1684</td><td style="border:1px solid #ddd; padding:4px;">1.2169</td><td style="border:1px solid #ddd; padding:4px;">1.000</td><td style="border:1px solid #ddd; padding:4px;">0.328</td><td style="border:1px solid #ddd; padding:4px;">0.266</td><td style="border:1px solid #ddd; padding:4px;">0.069</td><td style="border:1px solid #ddd; padding:4px;">0.651</td><td style="border:1px solid #ddd; padding:4px;">0.0006</td><td style="border:1px solid #ddd; padding:4px;">4.861</td></tr>
    </tbody>
  </table>
</div>
Test Results: Validity 1.000, SimSrc 0.331, SimTgt 0.261, dQED +0.068, PosDQED 0.825, Opt30 0.666, Opt40 0.274
Training finished.
"best_epoch": 68,
  "best_score": 4.9389569823507244,
  "best_valid_loss": 1.1714241946345905,


验证集评估：
{
  "validity": 1.0,
  "dQED": 0.0728414133042814,
  "SimSrc": 0.33056054677645613,
  "SimTgt": 0.26615034674035,
  "positive_dqed_rate": 0.827120822622108,
  "sim30_rate": 0.6812339331619537,
  "sim40_rate": 0.2699228791773779,
  "opt30_rate": 0.6606683804627249,
  "opt40_rate": 0.2609254498714653,
  "valid_count": 1556,
  "invalid_count": 0,
  "loss": 1.1714241946345905
}


测试集评估：
{
  "validity": 1.0,
  "dQED": 0.06790234655422303,
  "SimSrc": 0.3314666424439256,
  "SimTgt": 0.2609290724551484,
  "positive_dqed_rate": 0.8251928020565553,
  "sim30_rate": 0.6850899742930592,
  "sim40_rate": 0.2808483290488432,
  "opt30_rate": 0.6658097686375322,
  "opt40_rate": 0.2744215938303342,
  "valid_count": 1556,
  "invalid_count": 0,
  "loss": 1.1733037249740168
}
