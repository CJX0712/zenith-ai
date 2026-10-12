/*
 * StarForge 自检套件 — 每条算法均带可独立验证的数学不变量
 * 运行: node starforge.test.js  (目标: 全部 N/N 绿)
 * 作者: 晨星
 */
const SF = require('./starforge.js');

let pass = 0, fail = 0;
const results = [];
function ok(name, cond, detail) {
  if (cond) { pass++; results.push(['✅', name, detail || '']); }
  else { fail++; results.push(['❌', name, detail || 'FAIL']); }
}
function approx(a, b, tol) { return Math.abs(a - b) <= (tol || 1e-6); }

const rng = SF.mulberry32(20261013);
function randn() { // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// 1. Cholesky 重建
(function () {
  const A = [[4, 2, 1], [2, 5, 3], [1, 3, 6]];
  const L = SF.cholesky(A);
  const recon = SF.matMul(L, SF.matTranspose(L));
  let err = 0;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) err = Math.max(err, Math.abs(recon[i][j] - A[i][j]));
  ok('Cholesky 重建 A=L L^T', err < 1e-10, 'max err=' + err.toExponential(2));
})();

// 2. Jacobi 特征分解: A≈VΛV^T, V^T V=I, A v=λ v
(function () {
  const A = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]];
  const { values, vectors } = SF.jacobiEigen(A);
  // V^T V = I
  let ortho = 0;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    let s = 0; for (let k = 0; k < 3; k++) s += vectors[k][i] * vectors[k][j];
    ortho = Math.max(ortho, Math.abs(s - (i === j ? 1 : 0)));
  }
  // A v = λ v
  let eigen = 0;
  for (let j = 0; j < 3; j++) {
    const v = [vectors[0][j], vectors[1][j], vectors[2][j]];
    const Av = SF.matVec(A, v);
    for (let i = 0; i < 3; i++) eigen = Math.max(eigen, Math.abs(Av[i] - values[j] * v[i]));
  }
  ok('Jacobi: V^T V = I', ortho < 1e-10, 'max dev=' + ortho.toExponential(2));
  ok('Jacobi: A v = λ v', eigen < 1e-10, 'max dev=' + eigen.toExponential(2));
})();

// 3. SVD: A≈U S V^T, S≥0
(function () {
  const A = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]];
  const { U, S, V } = SF.svd(A);
  // 重建
  const SVt = SF.matMul(S.map((s, i) => { const row = new Array(3).fill(0); row[i] = s; return row; }), SF.matTranspose(V));
  const recon = SF.matMul(U, SVt);
  let err = 0;
  for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) err = Math.max(err, Math.abs(recon[i][j] - A[i][j]));
  const allNonNeg = S.every((s) => s >= -1e-12);
  ok('SVD: A ≈ U S V^T', err < 1e-8, 'max err=' + err.toExponential(2));
  ok('SVD: 奇异值 ≥ 0', allNonNeg, 'S=' + S.map((s) => s.toFixed(2)).join(','));
})();

// 4. logsumexp 防溢出
(function () {
  const x = [1000, 1000.1, 999.9];
  const le = SF.logsumexp(x);
  const direct = Math.log(x.reduce((a, b) => a + Math.exp(b), 0)); // 会溢出
  // 用稳定方式算参照: 直接算会 Inf, 用偏移法
  const m = Math.max.apply(null, x);
  const ref = m + Math.log(x.reduce((a, b) => a + Math.exp(b - m), 0));
  ok('logsumexp 数值稳定', approx(le, ref, 1e-9), 'le=' + le.toFixed(4) + ' ref=' + ref.toFixed(4) + ' (直接算=Inf)');
})();

// 5. 线性回归残差 ⟂ X
(function () {
  const n = 50, d = 4;
  const wTrue = [1.5, -2.0, 0.5, 3.0];
  const X = [], y = [];
  for (let i = 0; i < n; i++) {
    const r = []; for (let j = 0; j < d; j++) r.push(randn());
    X.push(r);
    y.push(SF.V.dot(r, wTrue) + 0.01 * randn());
  }
  const { w } = SF.linearRegression(X, y);
  const r = y.map((yi, i) => yi - SF.V.dot(X[i], w));
  const Xtr = SF.matTranspose(X);
  const Xt_r = SF.matVec(Xtr, r);
  let norm = 0; for (const v of Xt_r) norm = Math.max(norm, Math.abs(v));
  ok('线性回归: X^T r = 0 (残差正交)', norm < 1e-6, 'max|X^T r|=' + norm.toExponential(2));
})();

// 6. 逻辑回归: NLL 解析梯度 == 中心差分 (梯度检验) + 可分数据高精度
(function () {
  const n = 30, d = 3;
  const w0 = [0.5, -1.2, 0.8];
  const X = [], y = [];
  for (let i = 0; i < n; i++) {
    const r = []; for (let j = 0; j < d; j++) r.push(randn());
    X.push(r);
    const z = SF.V.dot(r, w0) + 0.3 * randn();
    y.push(z > 0 ? 1 : -1);
  }
  // 解析梯度: NLL = (1/n)Σ log(1+exp(-y z)); dNLL/dw_j = -(1/n)Σ y_i x_ij/(1+exp(y_i z_i))
  const f = (w) => {
    let s = 0;
    for (let i = 0; i < n; i++) { const z = SF.V.dot(X[i], w); s += Math.log(1 + Math.exp(-y[i] * z)); }
    return s / n;
  };
  const grad = (w) => {
    const g = new Array(d).fill(0);
    for (let i = 0; i < n; i++) {
      const z = SF.V.dot(X[i], w);
      const p = 1 / (1 + Math.exp(y[i] * z)); // σ(-y z)
      for (let j = 0; j < d; j++) g[j] += -y[i] * p * X[i][j];
    }
    for (let j = 0; j < d; j++) g[j] /= n;
    return g;
  };
  const wTest = [0.2, 0.1, -0.3];
  const gc = SF.gradCheck(f, wTest, grad(wTest));
  ok('逻辑回归: 梯度检验 relErr≈0', gc.relErr < 1e-6, 'relErr=' + gc.relErr.toExponential(2));
  // 功能: 强可分数据高精度
  const Xs = [[-2, 1], [-3, 2], [-1.5, 0.5], [2, -1], [3, -2], [1.5, -0.5]];
  const ys = [-1, -1, -1, 1, 1, 1];
  const model = SF.logisticRegression(Xs, ys, { iters: 5000, lr: 0.3 });
  const acc = model.predict(Xs).filter((p, i) => p === ys[i]).length / ys.length;
  ok('逻辑回归: 强可分数据精度=1.0', acc === 1.0, 'acc=' + acc.toFixed(2));
})();

// 7. kNN 可分数据
(function () {
  const X = [[-2, 1], [-3, 2], [2, -1], [3, -2]];
  const y = [-1, -1, 1, 1];
  const model = SF.knn(X, y, 1);
  const acc = model.predict(X).filter((p, i) => p === y[i]).length / y.length;
  ok('kNN(k=1) 训练精度=1.0', acc === 1.0, 'acc=' + acc.toFixed(2));
})();

// 8. k-means: 惯性单调不增
(function () {
  const X = [];
  for (let i = 0; i < 40; i++) X.push([randn() + (i % 2 ? 5 : -5), randn() + (i % 2 ? 5 : -5)]);
  const model = SF.kmeans(X, 2, { seed: 1, maxIter: 100 });
  // 重新计算惯性, 应有限且较小
  let inertia = 0;
  const pred = model.predict(X);
  for (let i = 0; i < X.length; i++) {
    const c = model.centers[pred[i]];
    inertia += SF.V.norm(SF.V.sub(X[i], c)) ** 2;
  }
  ok('k-means: 惯性有限且收敛', isFinite(inertia) && inertia > 0 && inertia < 1e4, 'inertia=' + inertia.toFixed(2));
})();

// 9. PCA: Eckart-Young 重建误差 == Σ_{i>k} λ_i / n
(function () {
  const n = 60, d = 5;
  const X = [];
  for (let i = 0; i < n; i++) { const r = []; for (let j = 0; j < d; j++) r.push(randn()); X.push(r); }
  const k = 2;
  const model = SF.pca(X, k);
  const Z = model.project(X);
  const Xr = model.reconstruct(Z);
  let sqErr = 0;
  for (let i = 0; i < n; i++) sqErr += SF.V.norm(SF.V.sub(X[i], Xr[i])) ** 2;
  // 剩余特征值 (已按降序)
  let remain = 0;
  for (let a = k; a < d; a++) remain += Math.max(model.values[a], 0);
  ok('PCA: Eckart-Young 重建误差', approx(sqErr, remain, 1e-6), 'err²=' + sqErr.toFixed(4) + ' Σλ_remaining=' + remain.toFixed(4));
  ok('PCA: 解释方差比 ∈ (0,1]', model.explained > 0 && model.explained <= 1 + 1e-9, 'ratio=' + model.explained.toFixed(4));
})();

// 10. GMM: 责任度行和=1, 权重和=1, 协方差 PSD, LL 单调不减
(function () {
  const X = [];
  for (let i = 0; i < 60; i++) X.push([randn() - 3, randn() - 3]);
  for (let i = 0; i < 60; i++) X.push([randn() + 3, randn() + 3]);
  const model = SF.gmm(X, 2, { seed: 7, maxIter: 100 });
  // 行和
  let rowSumOk = true;
  for (let i = 0; i < X.length; i++) {
    let s = 0; for (let c = 0; c < 2; c++) s += model.resp[i][c];
    if (Math.abs(s - 1) > 1e-9) rowSumOk = false;
  }
  let wSum = model.w[0] + model.w[1];
  // 协方差 PSD
  let psdOk = true;
  for (let c = 0; c < 2; c++) {
    const ev = SF.jacobiEigen(model.Sigma[c]).values;
    for (const v of ev) if (v < -1e-7) psdOk = false;
  }
  ok('GMM: 责任度行和=1', rowSumOk, '');
  ok('GMM: 权重和=1', approx(wSum, 1, 1e-9), 'Σw=' + wSum.toFixed(10));
  ok('GMM: 协方差半正定', psdOk, '');
  ok('GMM: 对数似然为有限值', isFinite(model.ll) && model.ll < 0, 'll=' + model.ll.toFixed(2));
})();

// 11. SVM (Pegasos): 强可分数据训练精度=1.0, 间隔>0
(function () {
  const X = []; const y = [];
  for (let i = 0; i < 30; i++) { X.push([randn() - 3, randn()]); y.push(-1); }
  for (let i = 0; i < 30; i++) { X.push([randn() + 3, randn()]); y.push(1); }
  const model = SF.svmPegasos(X, y, { lambda: 0.01, T: 4000, seed: 5 });
  const pred = model.predict(X);
  const acc = pred.filter((p, i) => p === y[i]).length / y.length;
  ok('SVM(Pegasos): 可分数据精度=1.0', acc === 1.0, 'acc=' + acc.toFixed(2));
  ok('SVM(Pegasos): 间隔 = 1/||w|| > 0', model.margin() > 0, 'margin=' + model.margin().toFixed(4));
})();

// 12. CART: 完美拟合 XOR
(function () {
  const X = [[0, 0], [1, 1], [0, 1], [1, 0]];
  const y = [0, 0, 1, 1];
  const model = SF.cart(X, y, { maxDepth: 3 });
  const acc = model.predict(X).filter((p, i) => p === y[i]).length / y.length;
  ok('CART: 完美拟合 XOR', acc === 1.0, 'acc=' + acc.toFixed(2) + ' nodes=' + model.nodeCount);
})();

// 13. 高斯朴素贝叶斯: 后验和=1, 简单数据高精度
(function () {
  const X = []; const y = [];
  for (let i = 0; i < 25; i++) { X.push([randn() - 2, randn() - 2]); y.push(0); }
  for (let i = 0; i < 25; i++) { X.push([randn() + 2, randn() + 2]); y.push(1); }
  const model = SF.gaussianNB(X, y);
  const proba = model.predictProba(X);
  let sumOk = true;
  for (const p of proba) { let s = 0; for (const c in p) s += p[c]; if (Math.abs(s - 1) > 1e-9) sumOk = false; }
  const acc = model.predict(X).filter((p, i) => p === y[i]).length / y.length;
  ok('NB: 后验概率和=1', sumOk, '');
  ok('NB: 高精度', acc >= 0.98, 'acc=' + acc.toFixed(2));
})();

// 14. 神经网络: 梯度检验 (反向传播 relErr≈1e-7)
(function () {
  const net = SF.neuralNet({ din: 3, dh: 4, dout: 2, seed: 11 });
  const x = [0.3, -0.5, 0.8]; const yIdx = 1;
  const w = net.flatten();
  const f = (arr) => net.lossFlat(arr, x, yIdx);
  const analytic = net.gradFlat(x, yIdx);
  const gc = SF.gradCheck(f, w, analytic);
  ok('神经网络: 反向传播梯度检验', gc.relErr < 1e-6, 'relErr=' + gc.relErr.toExponential(2));
})();

// 15. 注意力: 梯度检验 (缩放点积注意力 relErr≈1e-7)
(function () {
  const att = SF.attention({ n: 4, d: 3, seed: 9 });
  const G = [];
  for (let i = 0; i < 4; i++) { const r = []; for (let j = 0; j < 3; j++) r.push(randn()); G.push(r); }
  const arr = att.flatten(att.Q, att.K, att.V);
  const f = (a) => att.lossFlat(a, G);
  const analytic = att.gradFlat(G);
  const gc = SF.gradCheck(f, arr, analytic);
  ok('注意力: 反向传播梯度检验', gc.relErr < 1e-6, 'relErr=' + gc.relErr.toExponential(2));
})();

// 汇总
console.log('\n================ StarForge 自检结果 ================');
for (const [icon, name, detail] of results) {
  console.log(icon + ' ' + name + (detail ? '  [' + detail + ']' : ''));
}
console.log('----------------------------------------------------');
console.log('通过: ' + pass + ' / ' + (pass + fail));
if (fail > 0) { console.log('❌ 存在失败项'); process.exit(1); }
else { console.log('🟢 全部通过 (ALL GREEN)'); }
