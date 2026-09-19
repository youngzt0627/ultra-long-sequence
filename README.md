# 超长序列推荐实验

基于 PyTorch 的点击率预测实验，包含基础模型以及 SIM、ETA、TWIN 等长序列建模方案。

## 目录

- `base/`：基础序列推荐模型
- `sim/`：同类目双索引序列模型
- `eta/`：长短序列检索与注意力模型
- `twin/`：TWIN 长序列模型
- `data/`：数据预处理脚本

## 使用

安装 `torch`、`pandas`、`numpy`、`scikit-learn`，准备数据后从项目根目录运行，例如：

```bash
python base/base.py --data_dir ./data --device cpu
python sim/sim.py --data_dir ./data --sim_dir ./sim --device cpu
```

原始数据与生成的数据文件未包含在仓库中。

