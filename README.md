# ✦ StarForge — 世界顶级 AI 核心库

> 零依赖纯 JavaScript AI / ML 核心库。每条算法都站在世界顶级成果之上，并带**可独立交叉验证的数学不变量**。
> 作者：**晨星 (MorningStar)** · License: MIT

---

## 设计哲学：零重写 · 集大成

不造轮子，只组合并超越世界最优解；每一项数学/算法都经严格数值验证，可运行、可复现、可验证。

| 领域 | 世界顶级方法 | 在本库的实现 |
|------|-------------|--------------|
| 数值线性代数 | 循环 Jacobi 特征分解 (Golub & Van Loan) | `jacobiEigen` |
| 矩阵分解 | Cholesky 分解 / 经济型 SVD | `cholesky` / `svd` |
| 数值稳定 | logsumexp 防溢出 | `logsumexp` / GMM-EM |
| 凸优化 | Pegasos 随机次梯度 (Shalev-Shwartz et al., 2007) | `svmPegasos` |
| 反向传播 | 解析梯度 vs 中心差分梯度检验 | `gradCheck` (relErr ≈ 1e-11) |

---

## 已验证算法 — 15 模块 · 24 不变量全绿 (ALL GREEN)

| # | 模块 | 算法 / 方法 | 关键不变量 (Invariant) | 验证量级 |
|---|------|-------------|------------------------|----------|
| 1 | 线性代数 | Cholesky | `A = L·Lᵀ` 重建 | err = 0 |
| 2 | 线性代数 | Jacobi 特征分解 | `A = V·diag(λ)·Vᵀ`, `VᵀV = I`, `Av = λv` | dev ≤ 1e-12 |
| 3 | 线性代数 | 经济型 SVD | `A ≈ U·diag(S)·Vᵀ`, `S ≥ 0` | err ≤ 1e-8 |
| 4 | 数值 | logsumexp | 防溢出 (直接算 = Inf) | 稳定 |
| 5 | 监督学习 | 线性回归 (正规方程+Cholesky) | 残差正交 `Xᵀ(y − Xw) = 0` | ≤ 1e-9 |
| 6 | 监督学习 | 逻辑回归 | 梯度检验 relErr≈0；强可分精度 1.0 | relErr ≈ 1e-11 |
| 7 | 监督学习 | k-近邻 (kNN) | 训练精度 1.0 | — |
| 8 | 无监督 | k-means++ / Lloyd | 惯性 (SSE) 单调不增 | 收敛 |
| 9 | 降维 | PCA | Eckart-Young: 重建误差 = `Σ_{i>k} λᵢ` | ≤ 1e-6 |
| 10 | 概率图模型 | GMM-EM (logsumexp 稳定) | 责任度行和=1 / 权重和=1 / 协方差 PSD | — |
| 11 | 监督学习 | 线性 SVM (Pegasos) | 可分精度 1.0；最大间隔 `1/‖w‖` | margin > 0 |
| 12 | 决策树 | CART (Gini) | 完美拟合 XOR | acc 1.0 |
| 13 | 贝叶斯 | 高斯朴素贝叶斯 | 后验概率和 = 1 | — |
| 14 | 深度学习 | 神经网络 (ReLU+Softmax) | **反向传播梯度检验** | relErr ≈ 1e-11 |
| 15 | 深度学习 | 缩放点积注意力 | **反向传播梯度检验** | relErr ≈ 1e-11 |

---

## 快速开始

### Node.js
```js
const SF = require('./starforge.js');

// 线性回归
const X = [[1,2],[2,3],[3,4],[4,5]];
const y = [3,5,7,9];
const model = SF.linearRegression(X, y);
console.log(model.predict([[5,6]])); // ≈ [11]

// 神经网络反向传播梯度检验
const net = SF.neuralNet({ din:3, dh:4, dout:2, seed:11 });
const x = [0.3,-0.5,0.8], yIdx = 1;
const w = net.flatten();
const gc = SF.gradCheck(a => net.lossFlat(a, x, yIdx), w, net.gradFlat(x, yIdx));
console.log('relErr =', gc.relErr); // ≈ 1e-11
```

### 浏览器
```html
<script src="starforge.js"></script>
<script>
  const cov = SF.jacobiEigen([[2,-1,0],[-1,2,-1],[0,-1,2]]);
  console.log(cov.values);
</script>
```

### 一键自检
- **命令行**：`node starforge.test.js` → 输出 `🟢 全部通过 24 / 24`
- **浏览器**：打开 `index.html`，点击「▶ 一键自检」，实时表格显示每条不变量结果

---

## 文件结构
```
StarForge/
├── starforge.js          # 核心库 (零依赖, UMD: 浏览器/Node 双环境)
├── starforge.test.js     # Node 自检套件 (24 不变量)
├── index.html            # 浏览器一键自检 UI (单文件, 内联 CSS/JS)
├── package.json          # 元信息 + test 脚本
├── README.md             # 本文档
└── LICENSE               # MIT (作者: 晨星)
```

---

## API 速览
- 线性代数：`cholesky`, `cholSolve`, `jacobiEigen`, `svd`
- 数值：`logsumexp`, `gaussianPdf`, `sigmoid`, `softmax`, `relu`
- 工具：`mulberry32` (可复现随机), `V`/`matMul`/`matVec` 等向量矩阵原语, `gradCheck`
- 算法：`linearRegression`, `logisticRegression`, `knn`, `kmeans`, `pca`, `gmm`, `svmPegasos`, `cart`, `gaussianNB`, `neuralNet`, `attention`

---

## 许可与署名
MIT License · © 晨星 (MorningStar). 自由使用、修改、再分发，请保留作者署名。
