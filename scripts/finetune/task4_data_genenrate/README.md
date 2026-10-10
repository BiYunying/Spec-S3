# FG26 Task 4 数据生成

本目录按用户指定保留 `task4_data_genenrate` 拼写。单一入口 `generate.py`
生成 large-gap / small-gap 的配对、协议记录和 OT＋PCA 条件，运行时只依赖
`spec-s3/`。这是可重建的新协议，不承诺与历史配对及结果完全一致。

推荐在 `spec-s3/` 下使用一条微调命令（先激活 spectra 环境）：

```bash
# 已有 Task 3 GPT 时可跳过 warmup，直接提供它的 checkpoint
bash scripts/warmup.sh --decoder gpt
bash scripts/multiobjective.sh \
  --checkpoint experiments/warmup_gpt/last.ckpt --objective large_gap \
  --homo-threshold -7.63 --lumo-threshold -0.50 --device cuda:0
```

`multiobjective.sh` 自动执行 `generate.py --phase finetune`，顺序完成配对、
OT＋PCA 条件构建、选轮数和恢复初始权重后的最终微调。large/small-gap 共用
warmup，但分别构建条件、训练和保存 decoder；默认 `--objective both`。
Task 4 全流程仅支持 FG26，已移除 QM9S 分支。

默认权重写入 `datasets/FG26/task4_generated/<objective>/training/decoder_final.pt`。
`--output-dir` 指定整套数据/输出的父目录；`--log-dir` 单独指定训练输出目录
（both 时自动再分 large_gap/small_gap）。`--num-epochs`、`--batch-size`、`--lr`、
`--patience` 可覆盖默认训练参数。无显式阈值时仍使用训练集分位数。
训练完成后，直接把输出的 `config.yaml` 交给 `test/generate`、`test/evaluation`；
此命令不会自动生成评测候选。

以下为可选的分步数据准备入口，无需在单命令微调之前额外执行：

```bash
# 无需模型权重：先生成两个目标的配对和配置
python scripts/finetune/task4_data_genenrate/generate.py --phase prepare

# 之后用训练好的 FG26 GPT 构建条件缓存
python scripts/finetune/task4_data_genenrate/generate.py --phase build \
  --checkpoint experiments/warmup_gpt/last.ckpt --device cuda:0
```

这里的 `experiments/warmup_gpt/last.ckpt` 是运行
`bash scripts/warmup.sh --decoder gpt`
后得到的模型；不随公开目录附送。也可以指定自己的完整 FG26 GPT checkpoint。
纯预训练 checkpoint 没有训练好的 GPT/projector，不能直接代替。

一次生成全部文件，可用 `--phase all`（默认）及 `--checkpoint`。
**所有相对路径均相对于 `spec-s3/`**，与启动时的工作目录无关。
默认输出 `datasets/FG26/task4_generated/{large_gap,small_gap}/`，不会覆盖现有
`datasets/FG26/task4/`。重复运行请换 `--output-dir`；build 在 prepare 后执行。

```bash
python scripts/finetune/task4_data_genenrate/generate.py --phase all \
  --objective large_gap --checkpoint experiments/warmup_gpt/last.ckpt \
  --output-dir datasets/FG26/task4_new --anchor-count 40 \
  --val-draws 16 --test-draws 64 --pca-rank 12 --beta 0.15 --device cuda:0
```

## 输入与选择规则

- 默认光谱：`datasets/FG26/processed/fg26_uv_ir_raman_smiles_aligned.pt`。
- 默认真实 DFT 性质表：`datasets/FG26/fg26_property.csv`，包含
  `smiles,homo,lumo`，HOMO/LUMO 单位为 eV。按完整 InChIKey 对齐，不按行序猜测。
  原表含少数同分子标签冲突，默认沿用原 mapper 的最后一行，终端报告冲突数量。
  可用 `--duplicate-property-policy error` 要求立即报错。
- 默认划分：`datasets/FG26/splits.npz`。拒绝行号重叠；canonical 分子身份重复时
  优先保留 test，其次 val，再 train，避免同一身份出现在两组 A/B 查询中。
- large-gap：低 HOMO、高 LUMO；small-gap：高 HOMO、低 LUMO。
  阈值仅由保留下来的官方训练分子的真实性质分位数计算，默认分位数分别
  0.87625 / 0.87825。可用 `--quantile` 或成对的
  `--homo-threshold` / `--lumo-threshold` 覆盖；显式阈值限单个目标。
- D 是同时满足两个阈值的公开参考分子，默认取 40 个；不足时报错，不放宽规则。
  `--anchor-count 0` 使用全部合格 D。D 可来自任意官方 split，明确属于公开
  few-shot 训练资源，不能当作未见过的测试目标；保留 train/non-train D 分组，
  供现有 decoder 选轮次使用。
- A 仅满足 HOMO 条件，B 仅满足 LUMO 条件。val/test 的 A/B 分别只来自各自
  split。分别以 Morgan radius 2、2048 bits 的 Tanimoto 相似度，对每个 D
  做唯一父分子最大总相似度匹配，生成一组三元组。与历史多阶段配对不完全相同。
- 条件重建复用 `scripts/multiobjective.py`：原 FG26 三模态权重采样、余弦
  Hungarian OT（禁止分配回自己的 D）、可见 D 的 PCA 残差。数量、随机种子、
  PCA rank 和投影幅度均可配置。所有条件由指定 checkpoint 重新编码。

## 输出与下一步

| 输出 | 内容 |
|---|---|
| `known_dual_anchors.csv` | 公开 D 的身份、划分、真实性质和骨架 |
| `val_pairs.csv`, `test_pairs.csv` | 按 draw 展开的 A/B/D 配对；build 后补入 OT target 行号 |
| `val_conditions.npz`, `test_conditions.npz` | `operator_z`, `source_row_ids`, `target_row_ids`, 融合权重及 PCA basis |
| `val_conditions.json`, `test_conditions.json` | 编码 checkpoint 哈希、随机种子和 OT/PCA 参数 |
| `protocol_summary.json` | 固定 8 项：`objective`、`threshold_from`、`homo_min/max`、`lumo_min/max`、`anchors`、`anchor_splits`、`test_queries`、`generation_budget` |
| `config.yaml` | 唯一运行配置，包括构建所需采样次数及数据校验值 |

large-gap 用 `homo_max` / `lumo_min`；small-gap 用 `homo_min` / `lumo_max`。阈值单位为 eV，均包含边界。`threshold_from` 默认为 `train_dft`，手动指定阈值时为 `explicit`；仅旧输出无法恢复来源时记为 `unknown`。`test_queries` 为测试三元组数量，`generation_budget` 为生成次数，含无效及重复序列。prepare/build/all 使用同一记录格式，与历史目录字段一致，不再写入 `status` 或 `train_pairs`。

新输出不再创建 `decoder_training.json`。`protocol_summary.json` 仅用于查看，删除后仍能 build；运行参数全部读取 `config.yaml`。旧版 prepare 输出首次 build 时会从旧 summary 迁移缺少的字段，迁移后不再依赖它。

默认每个目标 40×16=640 条 val 和 40×64=2560 条 test 条件。
prepare 阶段尚未执行 OT，因此没有条件 NPZ 和 `transport_target_row_id`。

```bash
# 使用生成的配置微调 decoder（不再依赖历史 anchor decoder）
python scripts/train.py --task multiobjective \
  --conf datasets/FG26/task4_generated/large_gap/config.yaml

# 训练完成后生成候选
python test/generate/generate.py --task multiobjective \
  --conf datasets/FG26/task4_generated/large_gap/config.yaml
```

`--init-decoder` 可为后续微调指定已有 anchor decoder；省略则从完整 GPT 模型
内的 decoder 开始。新配置的 SMILES 随机化为 0，遵循仅 translation 训练启用
随机 SMILES 的约定。训练后实际选轮次与损失写入 `training/selection.json` 和
`selection.csv`，权重写入 `training/decoder_final.pt`。新生成的数据说明不是训练结果。
