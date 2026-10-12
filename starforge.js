/*
 * StarForge — 零依赖纯 JS 世界顶级 AI 核心库
 * 作者: 晨星 (MorningStar)  |  License: MIT
 *
 * 设计原则（集大成 / 零重写）:
 *   - 线性代数: 循环 Jacobi 特征分解 (Golub & Van Loan), Cholesky 分解
 *   - 数值稳定: logsumexp 防溢出 (GMM/EM, softmax)
 *   - 优化:     Pegasos 随机次梯度 (Shalev-Shwartz et al. 2007) 求解线性 SVM
 *   - 验证:     每个模块带可独立交叉验证的数学不变量 (解析解 vs 数值解, 梯度检验)
 *   - 零外部依赖, 浏览器 (window.SF) 与 Node (require) 双环境可用
 */
(function (global) {
  'use strict';

  // ---------------------------------------------------------------------------
  // 基础工具: 可复现随机数 (mulberry32)
  // ---------------------------------------------------------------------------
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ---------------------------------------------------------------------------
  // 向量 / 矩阵原语
  // ---------------------------------------------------------------------------
  const V = {
    zeros(n) { return new Array(n).fill(0); },
    add(a, b) { return a.map((x, i) => x + b[i]); },
    sub(a, b) { return a.map((x, i) => x - b[i]); },
    scale(a, s) { return a.map((x) => x * s); },
    dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; },
    norm(a) { return Math.sqrt(V.dot(a, a)); }
  };

  function matZeros(r, c) {
    const m = new Array(r);
    for (let i = 0; i < r; i++) m[i] = new Array(c).fill(0);
    return m;
  }
  function matIdentity(n) {
    const m = matZeros(n, n);
    for (let i = 0; i < n; i++) m[i][i] = 1;
    return m;
  }
  function matTranspose(A) {
    const r = A.length, c = A[0].length;
    const T = matZeros(c, r);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) T[j][i] = A[i][j];
    return T;
  }
  function matMul(A, B) {
    const r = A.length, k = B.length, c = B[0].length;
    const C = matZeros(r, c);
    for (let i = 0; i < r; i++) {
      for (let p = 0; p < k; p++) {
        const a = A[i][p];
        if (a === 0) continue;
        const Bp = B[p];
        const Ci = C[i];
        for (let j = 0; j < c; j++) Ci[j] += a * Bp[j];
      }
    }
    return C;
  }
  function matVec(A, x) {
    const r = A.length, c = A[0].length;
    const y = new Array(r);
    for (let i = 0; i < r; i++) {
      let s = 0;
      for (let j = 0; j < c; j++) s += A[i][j] * x[j];
      y[i] = s;
    }
    return y;
  }
  function vecMat(x, A) {
    const c = A[0].length, r = A.length;
    const y = new Array(c);
    for (let j = 0; j < c; j++) {
      let s = 0;
      for (let i = 0; i < r; i++) s += x[i] * A[i][j];
      y[j] = s;
    }
    return y;
  }

  // ---------------------------------------------------------------------------
  // 数值核心: Cholesky / Jacobi 特征分解 / SVD / logsumexp / 多元高斯
  // ---------------------------------------------------------------------------

  // Cholesky: A = L L^T, L 下三角. 不变量: L L^T == A
  function cholesky(A) {
    const n = A.length;
    const L = matZeros(n, n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = 0;
        for (let k = 0; k < j; k++) sum += L[i][k] * L[j][k];
        if (i === j) {
          const d = A[i][i] - sum;
          if (d <= 1e-12) throw new Error('矩阵非正定, Cholesky 失败');
          L[i][j] = Math.sqrt(d);
        } else {
          L[i][j] = (A[i][j] - sum) / L[j][j];
        }
      }
    }
    return L;
  }

  // 解 L L^T x = b (L 下三角)
  function cholSolve(L, b) {
    const n = b.length;
    const y = new Array(n);
    for (let i = 0; i < n; i++) {
      let s = b[i];
      for (let k = 0; k < i; k++) s -= L[i][k] * y[k];
      y[i] = s / L[i][i];
    }
    const x = new Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let s = y[i];
      for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
      x[i] = s / L[i][i];
    }
    return x;
  }

  // 循环 Jacobi 特征分解 (对称阵). 不变量: A == V diag(λ) V^T, V^T V == I, A v_i = λ_i v_i
  function jacobiEigen(Ain, maxSweeps, tol) {
    maxSweeps = maxSweeps || 100;
    tol = tol || 1e-13;
    const n = Ain.length;
    const A = Ain.map((r) => r.slice());
    const V = matIdentity(n);
    for (let sweep = 0; sweep < maxSweeps; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++)
        for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
      if (off < tol) break;
      for (let p = 0; p < n; p++) {
        for (let q = p + 1; q < n; q++) {
          if (Math.abs(A[p][q]) < 1e-300) continue;
          const app = A[p][p], aqq = A[q][q], apq = A[p][q];
          const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
          const c = Math.cos(phi), s = Math.sin(phi);
          // 左乘旋转: 更新列 p, q
          for (let i = 0; i < n; i++) {
            const aip = A[i][p], aiq = A[i][q];
            A[i][p] = c * aip - s * aiq;
            A[i][q] = s * aip + c * aiq;
          }
          // 右乘旋转: 更新行 p, q (此时 A[i][p],A[i][q] 已为 AJ)
          for (let j = 0; j < n; j++) {
            const apj = A[p][j], aqj = A[q][j];
            A[p][j] = c * apj - s * aqj;
            A[q][j] = s * apj + c * aqj;
          }
          // 累积特征向量 V = V J
          for (let i = 0; i < n; i++) {
            const vip = V[i][p], viq = V[i][q];
            V[i][p] = c * vip - s * viq;
            V[i][q] = s * vip + c * viq;
          }
        }
      }
    }
    const values = new Array(n);
    for (let i = 0; i < n; i++) values[i] = A[i][i];
    return { values, vectors: V }; // 列向量为特征向量
  }

  // 经济型 SVD: A(m×n) = U(m×n) diag(S) V^T(n×n). 不变量: A ≈ U diag(S) V^T, S≥0
  function svd(A) {
    const m = A.length, n = A[0].length;
    const ATA = matMul(matTranspose(A), A);
    const { values, vectors } = jacobiEigen(ATA);
    const S = values.map((v) => Math.sqrt(Math.max(v, 0)));
    const Vm = vectors;
    const Um = matZeros(m, n);
    for (let j = 0; j < n; j++) {
      if (S[j] > 1e-12) {
        for (let i = 0; i < m; i++) {
          let sum = 0;
          for (let t = 0; t < n; t++) sum += A[i][t] * Vm[t][j];
          Um[i][j] = sum / S[j];
        }
      }
    }
    return { U: Um, S, V: Vm };
  }

  // logsumexp — 防溢出. 不变量: exp(logsumexp(x)) ≈ Σ exp(x)
  function logsumexp(a) {
    let m = -Infinity;
    for (const x of a) if (x > m) m = x;
    let s = 0;
    for (const x of a) s += Math.exp(x - m);
    return m + Math.log(s);
  }

  // 多元高斯概率密度 N(x | μ, Σ). 用 Cholesky, 数值稳定
  function gaussianPdf(x, mu, Sigma) {
    const n = x.length;
    const L = cholesky(Sigma);
    let logdet = 0;
    for (let i = 0; i < n; i++) logdet += 2 * Math.log(L[i][i]);
    const d = x.map((xi, i) => xi - mu[i]);
    const z = new Array(n);
    for (let i = 0; i < n; i++) {
      let s = d[i];
      for (let k = 0; k < i; k++) s -= L[i][k] * z[k];
      z[i] = s / L[i][i];
    }
    let q = 0;
    for (let i = 0; i < n; i++) q += z[i] * z[i];
    return Math.exp(-0.5 * (q + logdet + n * Math.log(2 * Math.PI)));
  }

  // ---------------------------------------------------------------------------
  // 机器学习算法 (均带可验证不变量)
  // ---------------------------------------------------------------------------

  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }
  function softmax(x) {
    const m = Math.max.apply(null, x);
    const ex = x.map((v) => Math.exp(v - m));
    const s = ex.reduce((a, b) => a + b, 0);
    return ex.map((v) => v / s);
  }
  function relu(z) { return z > 0 ? z : 0; }

  // 线性回归: 闭式解 (正规方程 + Cholesky). 不变量: X^T (y - Xw) ≈ 0
  function linearRegression(X, y, ridge) {
    ridge = ridge || 1e-9;
    const n = X.length, d = X[0].length;
    const XtX = matMul(matTranspose(X), X);
    const Xty = matVec(matTranspose(X), y);
    for (let i = 0; i < d; i++) XtX[i][i] += ridge;
    const L = cholesky(XtX);
    const w = cholSolve(L, Xty);
    return {
      w,
      predict(Xn) { return Xn.map((r) => V.dot(r, w)); }
    };
  }

  // 逻辑回归 (y ∈ {-1,1}). 不变量: 解析梯度 == 中心差分梯度 (梯度检验)
  function logisticRegression(X, y, opts) {
    opts = opts || {};
    const n = X.length, d = X[0].length;
    const lr = opts.lr || 0.1, iters = opts.iters || 3000;
    let w = new Array(d).fill(0);
    for (let it = 0; it < iters; it++) {
      const g = new Array(d).fill(0);
      for (let i = 0; i < n; i++) {
        const z = V.dot(X[i], w);
        // NLL = (1/n)Σ log(1+exp(-y z)); dNLL/dw_j = -(1/n)Σ y_i x_ij σ(-y_i z_i)
        // 注意: σ(-y z) = 1/(1+exp(y z))  (不是 1/(1+exp(-y z)) = σ(y z))
        const p = 1 / (1 + Math.exp(y[i] * z));
        const err = -y[i] * p; // = -y σ(-y z)
        for (let j = 0; j < d; j++) g[j] += err * X[i][j];
      }
      for (let j = 0; j < d; j++) w[j] -= lr * (g[j] / n);
    }
    return {
      w,
      predict(Xn) { return Xn.map((r) => (V.dot(r, w) >= 0 ? 1 : -1)); }
    };
  }

  // k 近邻 (暴力). 距离: 欧氏
  function knn(X, y, k) {
    return {
      fit() {},
      predict(Xn) {
        return Xn.map((q) => {
          const dist = X.map((r, i) => ({ i, d: V.norm(V.sub(r, q)) }));
          dist.sort((a, b) => a.d - b.d);
          const top = dist.slice(0, k);
          const cnt = {};
          for (const t of top) {
            const lab = y[t.i];
            cnt[lab] = (cnt[lab] || 0) + 1;
          }
          let best = null, bc = -1;
          for (const key in cnt) if (cnt[key] > bc) { bc = cnt[key]; best = +key; }
          return best;
        });
      }
    };
  }

  // k-means++ 初始化 + Lloyd 迭代. 不变量: 惯性 (SSE) 单调不增
  function kmeans(X, k, opts) {
    opts = opts || {};
    const n = X.length, d = X[0].length;
    const rng = mulberry32(opts.seed || 42);
    const maxIter = opts.maxIter || 300;
    // k-means++
    const centers = [X[Math.floor(rng() * n)].slice()];
    const dist2 = new Array(n).fill(Infinity);
    for (let c = 1; c < k; c++) {
      let tot = 0;
      for (let i = 0; i < n; i++) {
        const dd = V.norm(V.sub(X[i], centers[centers.length - 1])) ** 2;
        if (dd < dist2[i]) dist2[i] = dd;
        tot += dist2[i];
      }
      let r = rng() * tot, idx = 0;
      for (let i = 0; i < n; i++) { r -= dist2[i]; if (r <= 0) { idx = i; break; } }
      centers.push(X[idx].slice());
    }
    let assign = new Array(n);
    let inertia = Infinity;
    for (let iter = 0; iter < maxIter; iter++) {
      let newInertia = 0;
      for (let i = 0; i < n; i++) {
        let best = 0, bd = Infinity;
        for (let c = 0; c < k; c++) {
          const dd = V.norm(V.sub(X[i], centers[c])) ** 2;
          if (dd < bd) { bd = dd; best = c; }
        }
        assign[i] = best;
        newInertia += bd;
      }
      if (newInertia > inertia - 1e-9) { inertia = newInertia; break; }
      inertia = newInertia;
      const sum = Array.from({ length: k }, () => new Array(d).fill(0));
      const cnt = new Array(k).fill(0);
      for (let i = 0; i < n; i++) {
        const c = assign[i]; cnt[c]++;
        for (let j = 0; j < d; j++) sum[c][j] += X[i][j];
      }
      for (let c = 0; c < k; c++)
        for (let j = 0; j < d; j++) centers[c][j] = cnt[c] ? sum[c][j] / cnt[c] : centers[c][j];
    }
    return {
      centers, assign,
      inertia,
      predict(Xn) {
        return Xn.map((q) => {
          let best = 0, bd = Infinity;
          for (let c = 0; c < k; c++) {
            const dd = V.norm(V.sub(q, centers[c])) ** 2;
            if (dd < bd) { bd = dd; best = c; }
          }
          return best;
        });
      }
    };
  }

  // PCA (协方差特征分解). 不变量: 秩-k 重建误差 == Σ_{i>k} λ_i (Eckart-Young)
  function pca(X, k, opts) {
    opts = opts || {};
    const n = X.length, d = X[0].length;
    const ridge = opts.ridge || 0;
    const mean = new Array(d).fill(0);
    for (const r of X) for (let j = 0; j < d; j++) mean[j] += r[j];
    for (let j = 0; j < d; j++) mean[j] /= n;
    const Xc = X.map((r) => r.map((v, j) => v - mean[j]));
    let cov = matMul(matTranspose(Xc), Xc);
    for (let i = 0; i < d; i++) cov[i][i] += ridge;
    const { values, vectors } = jacobiEigen(cov);
    const idx = Array.from({ length: d }, (_, i) => i).sort((a, b) => values[b] - values[a]);
    const comps = idx.slice(0, k).map((i) => vectors.map((row) => row[i]));
    const totalVar = values.reduce((a, b) => a + Math.max(b, 0), 0);
    const explained = idx.slice(0, k).reduce((a, i) => a + Math.max(values[i], 0), 0) / totalVar;
    return {
      mean, components: comps, values: idx.map((i) => values[i]), explained,
      project(Xn) {
        return Xn.map((r) => {
          const rc = r.map((v, j) => v - mean[j]);
          return comps.map((c) => V.dot(rc, c));
        });
      },
      reconstruct(Z) {
        return Z.map((z) => {
          const v = mean.slice();
          for (let a = 0; a < k; a++)
            for (let j = 0; j < d; j++) v[j] += z[a] * comps[a][j];
          return v;
        });
      }
    };
  }

  // 高斯混合模型 EM (logsumexp 稳定). 不变量: 责任度行和=1, 权重和=1, 协方差 PSD, LL 单调不减
  function gmm(X, k, opts) {
    opts = opts || {};
    const n = X.length, d = X[0].length;
    const rng = mulberry32(opts.seed || 7);
    const maxIter = opts.maxIter || 200;
    const minCov = opts.minCov || 1e-6;
    // 初始化: kmeans++ 取均值, 全局协方差
    const km = kmeans(X, k, { seed: opts.seed || 7 });
    let mu = km.centers.map((c) => c.slice());
    let Sigma = new Array(k);
    const globalCov = matZeros(d, d);
    const mean = new Array(d).fill(0);
    for (const r of X) for (let j = 0; j < d; j++) mean[j] += r[j];
    for (let j = 0; j < d; j++) mean[j] /= n;
    for (const r of X) for (let a = 0; a < d; a++) for (let b = 0; b < d; b++)
      globalCov[a][b] += (r[a] - mean[a]) * (r[b] - mean[b]);
    for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) globalCov[a][b] /= n;
    for (let c = 0; c < k; c++) {
      Sigma[c] = globalCov.map((row) => row.slice());
      for (let i = 0; i < d; i++) Sigma[c][i][i] += minCov;
    }
    let w = new Array(k).fill(1 / k);
    const resp = matZeros(n, k);
    let ll = -Infinity;

    function logProb(i, c) {
      const x = X[i];
      const L = cholesky(Sigma[c]);
      let logdet = 0;
      for (let t = 0; t < d; t++) logdet += 2 * Math.log(L[t][t]);
      const diff = x.map((xi, t) => xi - mu[c][t]);
      const z = new Array(d);
      for (let t = 0; t < d; t++) {
        let s = diff[t];
        for (let u = 0; u < t; u++) s -= L[t][u] * z[u];
        z[t] = s / L[t][t];
      }
      let q = 0;
      for (let t = 0; t < d; t++) q += z[t] * z[t];
      return -0.5 * (q + logdet + d * Math.log(2 * Math.PI));
    }

    for (let iter = 0; iter < maxIter; iter++) {
      // E-step
      let newLl = 0;
      for (let i = 0; i < n; i++) {
        const logw = w.map((wc, c) => Math.log(wc) + logProb(i, c));
        const lse = logsumexp(logw);
        newLl += lse;
        for (let c = 0; c < k; c++) resp[i][c] = Math.exp(logw[c] - lse);
      }
      if (newLl < ll - 1e-8) break; // 数值保护
      ll = newLl;
      // M-step
      const Nc = new Array(k).fill(0);
      for (let i = 0; i < n; i++) for (let c = 0; c < k; c++) Nc[c] += resp[i][c];
      for (let c = 0; c < k; c++) {
        w[c] = (Nc[c] + 1e-12) / n;
        const m = new Array(d).fill(0);
        for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) m[j] += resp[i][c] * X[i][j];
        for (let j = 0; j < d; j++) m[j] /= Nc[c] || 1e-12;
        mu[c] = m;
        const S = matZeros(d, d);
        for (let i = 0; i < n; i++) {
          const diff = X[i].map((xi, j) => xi - m[j]);
          for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) S[a][b] += resp[i][c] * diff[a] * diff[b];
        }
        for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) S[a][b] /= Nc[c] || 1e-12;
        for (let a = 0; a < d; a++) S[a][a] += minCov;
        Sigma[c] = S;
      }
    }
    return { w, mu, Sigma, ll, resp, predict(Xn) {
      return Xn.map((q) => {
        let best = 0, bv = -Infinity;
        for (let c = 0; c < k; c++) {
          const logw = Math.log(w[c]) + (function () {
            const L = cholesky(Sigma[c]);
            let logdet = 0; for (let t = 0; t < d; t++) logdet += 2 * Math.log(L[t][t]);
            const diff = q.map((xi, t) => xi - mu[c][t]);
            const z = new Array(d);
            for (let t = 0; t < d; t++) { let s = diff[t]; for (let u = 0; u < t; u++) s -= L[t][u] * z[u]; z[t] = s / L[t][t]; }
            let qd = 0; for (let t = 0; t < d; t++) qd += z[t] * z[t];
            return -0.5 * (qd + logdet + d * Math.log(2 * Math.PI));
          })();
          if (logw > bv) { bv = logw; best = c; }
        }
        return best;
      });
    } };
  }

  // 线性 SVM — Pegasos 随机次梯度 (Shalev-Shwartz 2007). 不变量: 可分数据训练精度高, 间隔 = 1/||w||
  function svmPegasos(X, y, opts) {
    opts = opts || {};
    const n = X.length, d = X[0].length;
    const lambda = opts.lambda || 0.01;
    const T = opts.T || 3000;
    const seed = opts.seed || 42;
    const rng = mulberry32(seed);
    let w = new Array(d).fill(0);
    for (let t = 1; t <= T; t++) {
      const i = Math.floor(rng() * n);
      const eta = 1 / (lambda * t);
      const score = V.dot(X[i], w);
      if (y[i] * score < 1) {
        for (let j = 0; j < d; j++) w[j] = (1 - eta * lambda) * w[j] + eta * y[i] * X[i][j];
      } else {
        for (let j = 0; j < d; j++) w[j] *= (1 - eta * lambda);
      }
    }
    return {
      w,
      margin() { return 1 / (V.norm(w) || 1e-12); },
      predict(Xn) { return Xn.map((r) => (V.dot(r, w) >= 0 ? 1 : -1)); }
    };
  }

  // 决策树 CART (Gini 二分). 不变量: 每次分裂 Gini 不增, 充分深度可完美拟合训练集
  function cart(X, y, opts) {
    opts = opts || {};
    const maxDepth = opts.maxDepth || 8;
    const minSplit = opts.minSplit || 2;
    let nodeCount = 0;

    function gini(counts, total) {
      if (total === 0) return 0;
      let s = 0;
      for (const k in counts) { const p = counts[k] / total; s += p * p; }
      return 1 - s;
    }
    function build(rows, depth) {
      nodeCount++;
      const labels = rows.map((r) => r.y);
      const counts = {};
      for (const l of labels) counts[l] = (counts[l] || 0) + 1;
      const majority = Object.keys(counts).sort((a, b) => counts[b] - counts[a])[0];
      if (depth >= maxDepth || rows.length < minSplit || Object.keys(counts).length === 1) {
        return { leaf: true, label: +majority, counts };
      }
      const d = rows[0].x.length;
      let best = null;
      for (let f = 0; f < d; f++) {
        const vals = rows.map((r) => r.x[f]).sort((a, b) => a - b);
        for (let s = 0; s < vals.length - 1; s++) {
          const thr = (vals[s] + vals[s + 1]) / 2;
          const left = { c: {}, n: 0 }, right = { c: {}, n: 0 };
          for (const r of rows) {
            if (r.x[f] <= thr) { left.c[r.y] = (left.c[r.y] || 0) + 1; left.n++; }
            else { right.c[r.y] = (right.c[r.y] || 0) + 1; right.n++; }
          }
          if (left.n === 0 || right.n === 0) continue;
          const g = (left.n * gini(left.c, left.n) + right.n * gini(right.c, right.n)) / rows.length;
          if (!best || g < best.g) best = { f, thr, g, left, right };
        }
      }
      if (!best) return { leaf: true, label: +majority, counts };
      const lRows = rows.filter((r) => r.x[best.f] <= best.thr).map((r) => ({ x: r.x, y: r.y }));
      const rRows = rows.filter((r) => r.x[best.f] > best.thr).map((r) => ({ x: r.x, y: r.y }));
      return {
        leaf: false, f: best.f, thr: best.thr, gini: best.g,
        left: build(lRows, depth + 1), right: build(rRows, depth + 1)
      };
    }
    const rows = X.map((x, i) => ({ x, y: y[i] }));
    const tree = build(rows, 0);
    function traverse(node, x) {
      if (node.leaf) return node.label;
      return traverse(x[node.f] <= node.thr ? node.left : node.right, x);
    }
    return {
      tree, nodeCount,
      predict(Xn) { return Xn.map((x) => traverse(tree, x)); }
    };
  }

  // 高斯朴素贝叶斯. 不变量: 后验概率和=1
  function gaussianNB(X, y) {
    const classes = [...new Set(y)];
    const d = X[0].length;
    const stats = {};
    for (const c of classes) {
      const rows = X.filter((_, i) => y[i] === c);
      const mean = new Array(d).fill(0), varr = new Array(d).fill(0);
      for (const r of rows) for (let j = 0; j < d; j++) mean[j] += r[j];
      for (let j = 0; j < d; j++) mean[j] /= rows.length;
      for (const r of rows) for (let j = 0; j < d; j++) varr[j] += (r[j] - mean[j]) ** 2;
      for (let j = 0; j < d; j++) varr[j] = (varr[j] / rows.length) + 1e-9;
      stats[c] = { mean, varr, prior: rows.length / X.length };
    }
    return {
      classes, stats,
      predictProba(Xn) {
        return Xn.map((x) => {
          const logp = {};
          let m = -Infinity;
          for (const c of classes) {
            const s = stats[c];
            let lp = Math.log(s.prior);
            for (let j = 0; j < d; j++) {
              const diff = x[j] - s.mean[j];
              lp += -0.5 * (Math.log(2 * Math.PI * s.varr[j]) + (diff * diff) / s.varr[j]);
            }
            logp[c] = lp; if (lp > m) m = lp;
          }
          const ex = {};
          let sum = 0;
          for (const c of classes) { ex[c] = Math.exp(logp[c] - m); sum += ex[c]; }
          const out = {};
          for (const c of classes) out[c] = ex[c] / sum;
          return out;
        });
      },
      predict(Xn) {
        return this.predictProba(Xn).map((p) => {
          let best = classes[0], bv = -1;
          for (const c of classes) if (p[c] > bv) { bv = p[c]; best = c; }
          return best;
        });
      }
    };
  }

  // ---------------------------------------------------------------------------
  // 神经网络 (ReLU + Softmax 交叉熵) — 手写反向传播. 不变量: 梯度检验 relErr ≈ 1e-7
  // ---------------------------------------------------------------------------
  function neuralNet(opts) {
    opts = opts || {};
    const din = opts.din, dh = opts.dh, dout = opts.dout, seed = opts.seed || 1;
    const rng = mulberry32(seed);
    const rand = (a, b) => (rng() - 0.5) * (b - a);
    // 参数
    const W1 = matZeros(dh, din), b1 = new Array(dh).fill(0);
    const W2 = matZeros(dout, dh), b2 = new Array(dout).fill(0);
    for (let i = 0; i < dh; i++) for (let j = 0; j < din; j++) W1[i][j] = rand(-1, 1);
    for (let i = 0; i < dout; i++) for (let j = 0; j < dh; j++) W2[i][j] = rand(-1, 1);

    function forward(x, yIdx) {
      const z1 = new Array(dh);
      for (let i = 0; i < dh; i++) z1[i] = V.dot(W1[i], x) + b1[i];
      const a1 = z1.map(relu);
      const z2 = new Array(dout);
      for (let i = 0; i < dout; i++) z2[i] = V.dot(W2[i], a1) + b2[i];
      const p = softmax(z2);
      const loss = -Math.log(p[yIdx] + 1e-12);
      return { z1, a1, z2, p, loss };
    }

    // 解析梯度 (单样本)
    function backward(x, yIdx) {
      const { a1, z1, z2, p } = forward(x, yIdx);
      const dz2 = p.slice(); dz2[yIdx] -= 1;
      const dW2 = matZeros(dout, dh), db2 = dz2.slice();
      for (let i = 0; i < dout; i++) for (let j = 0; j < dh; j++) dW2[i][j] = dz2[i] * a1[j];
      const da1 = new Array(dh);
      for (let j = 0; j < dh; j++) {
        let s = 0; for (let i = 0; i < dout; i++) s += W2[i][j] * dz2[i]; da1[j] = s;
      }
      const dz1 = da1.map((v, i) => (z1[i] > 0 ? v : 0));
      const dW1 = matZeros(dh, din), db1 = dz1.slice();
      for (let i = 0; i < dh; i++) for (let j = 0; j < din; j++) dW1[i][j] = dz1[i] * x[j];
      return { dW1, db1, dW2, db2 };
    }

    // 扁平化工具 (用于梯度检验)
    function flatten() {
      const arr = [];
      for (let i = 0; i < dh; i++) for (let j = 0; j < din; j++) arr.push(W1[i][j]);
      for (let i = 0; i < dh; i++) arr.push(b1[i]);
      for (let i = 0; i < dout; i++) for (let j = 0; j < dh; j++) arr.push(W2[i][j]);
      for (let i = 0; i < dout; i++) arr.push(b2[i]);
      return arr;
    }
    function fromFlat(arr) {
      let p = 0;
      for (let i = 0; i < dh; i++) for (let j = 0; j < din; j++) W1[i][j] = arr[p++];
      for (let i = 0; i < dh; i++) b1[i] = arr[p++];
      for (let i = 0; i < dout; i++) for (let j = 0; j < dh; j++) W2[i][j] = arr[p++];
      for (let i = 0; i < dout; i++) b2[i] = arr[p++];
    }
    function lossFlat(arr, x, yIdx) {
      fromFlat(arr);
      return forward(x, yIdx).loss;
    }
    function gradFlat(x, yIdx) {
      const g = backward(x, yIdx);
      const arr = [];
      for (let i = 0; i < dh; i++) for (let j = 0; j < din; j++) arr.push(g.dW1[i][j]);
      for (let i = 0; i < dh; i++) arr.push(g.db1[i]);
      for (let i = 0; i < dout; i++) for (let j = 0; j < dh; j++) arr.push(g.dW2[i][j]);
      for (let i = 0; i < dout; i++) arr.push(g.db2[i]);
      return arr;
    }

    // 训练 (SGD)
    function train(X, y, epochs, lr) {
      lr = lr || 0.1; epochs = epochs || 200;
      const n = X.length;
      for (let e = 0; e < epochs; e++) {
        const i = Math.floor(rng() * n);
        const g = backward(X[i], y[i]);
        for (let a = 0; a < dh; a++) for (let b = 0; b < din; b++) W1[a][b] -= lr * g.dW1[a][b];
        for (let a = 0; a < dh; a++) b1[a] -= lr * g.db1[a];
        for (let a = 0; a < dout; a++) for (let b = 0; b < dh; b++) W2[a][b] -= lr * g.dW2[a][b];
        for (let a = 0; a < dout; a++) b2[a] -= lr * g.db2[a];
      }
    }

    return {
      forward, backward, flatten, fromFlat, lossFlat, gradFlat, train,
      predict(Xn) { return Xn.map((x) => { const f = forward(x, 0); let best = 0, bv = -1; for (let i = 0; i < dout; i++) if (f.p[i] > bv) { bv = f.p[i]; best = i; } return best; }); }
    };
  }

  // ---------------------------------------------------------------------------
  // 注意力机制 — 缩放点积注意力 (前向 + 反向). 不变量: 梯度检验 relErr ≈ 1e-7
  // ---------------------------------------------------------------------------
  function attention(opts) {
    opts = opts || {};
    const n = opts.n, d = opts.d, seed = opts.seed || 3;
    const rng = mulberry32(seed);
    const rand = () => (rng() - 0.5) * 0.4;
    // 参数矩阵 Q,K,V (n×d)
    const Q = matZeros(n, d), K = matZeros(n, d), Vm = matZeros(n, d);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) { Q[i][j] = rand(); K[i][j] = rand(); Vm[i][j] = rand(); }

    function forward(Qm, Km, Vmat) {
      const scale = Math.sqrt(d);
      const scores = matZeros(n, n);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) scores[i][j] = V.dot(Qm[i], Km[j]) / scale;
      const A = scores.map((row) => softmax(row));
      const O = matZeros(n, d);
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) {
        let s = 0; for (let k = 0; k < n; k++) s += A[i][k] * Vmat[k][j]; O[i][j] = s;
      }
      return { scores, A, O };
    }

    // 反向: 给定上游梯度 G (n×d), 求 dQ,dK,dV
    function backward(Qm, Km, Vmat, G) {
      const { scores, A } = forward(Qm, Km, Vmat);
      const scale = Math.sqrt(d);
      const dV = matZeros(n, d);
      for (let k = 0; k < n; k++) for (let j = 0; j < d; j++) {
        let s = 0; for (let i = 0; i < n; i++) s += A[i][k] * G[i][j]; dV[k][j] = s;
      }
      // dA = G V^T
      const dA = matZeros(n, n);
      for (let i = 0; i < n; i++) for (let k = 0; k < n; k++) dA[i][k] = V.dot(G[i], Vmat[k]);
      // dscores = dA ⊙ A - A (Σ dA)  (softmax 行向量 Jacobian)
      const dscores = matZeros(n, n);
      for (let i = 0; i < n; i++) {
        let sum = 0; for (let k = 0; k < n; k++) sum += dA[i][k] * A[i][k];
        for (let k = 0; k < n; k++) dscores[i][k] = A[i][k] * (dA[i][k] - sum);
      }
      const dQ = matZeros(n, d), dK = matZeros(n, d);
      // dQ = dscores · K / scale   (对每个输出维度 j, 沿 k 累加)
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) {
        let sq = 0;
        for (let k = 0; k < n; k++) sq += dscores[i][k] * Km[k][j];
        dQ[i][j] = sq / scale;
      }
      // dK = dscores^T · Q / scale
      for (let k = 0; k < n; k++) for (let j = 0; j < d; j++) {
        let s = 0; for (let i = 0; i < n; i++) s += dscores[i][k] * Qm[i][j];
        dK[k][j] = s / scale;
      }
      return { dQ, dK, dV };
    }

    // 扁平化 (Q,K,V 拼接) 用于梯度检验, 损失 = Σ O⊙G
    function flatten(Qm, Km, Vmat) {
      const arr = [];
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) arr.push(Qm[i][j]);
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) arr.push(Km[i][j]);
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) arr.push(Vmat[i][j]);
      return arr;
    }
    function lossFlat(arr, G) {
      const Qm = matZeros(n, d), Km = matZeros(n, d), Vmat = matZeros(n, d);
      let p = 0;
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) Qm[i][j] = arr[p++];
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) Km[i][j] = arr[p++];
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) Vmat[i][j] = arr[p++];
      const { O } = forward(Qm, Km, Vmat);
      let s = 0; for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) s += O[i][j] * G[i][j];
      return s;
    }
    function gradFlat(G) {
      // 用解析反向, 但需相对扁平参数. 构造当前 Q,K,V 的 dQ,dK,dV 再拼接
      const Gm = G;
      const { dQ, dK, dV } = backward(Q, K, Vm, Gm);
      const arr = [];
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) arr.push(dQ[i][j]);
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) arr.push(dK[i][j]);
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) arr.push(dV[i][j]);
      return arr;
    }

    return { forward, backward, flatten, lossFlat, gradFlat, get Q() { return Q; }, get K() { return K; }, get V() { return Vm; } };
  }

  // ---------------------------------------------------------------------------
  // 通用梯度检验 (中心差分 vs 解析梯度) — 验证总纲的核心工具
  // ---------------------------------------------------------------------------
  function gradCheck(f, w, analytic, eps) {
    eps = eps || 1e-5;
    const num = w.map((_, i) => 0);
    for (let i = 0; i < w.length; i++) {
      const wp = w.slice(); wp[i] += eps;
      const wm = w.slice(); wm[i] -= eps;
      num[i] = (f(wp) - f(wm)) / (2 * eps);
    }
    let numNorm = 0, anNorm = 0, diff = 0;
    for (let i = 0; i < w.length; i++) {
      diff += Math.abs(num[i] - analytic[i]);
      numNorm += num[i] * num[i];
      anNorm += analytic[i] * analytic[i];
    }
    const relErr = diff / (Math.sqrt(numNorm) + Math.sqrt(anNorm) + 1e-12);
    return { relErr, num, analytic };
  }

  // ---------------------------------------------------------------------------
  // 导出
  // ---------------------------------------------------------------------------
  const SF = {
    version: '1.0.0',
    author: '晨星 (MorningStar)',
    mulberry32, V, matZeros, matIdentity, matTranspose, matMul, matVec, vecMat,
    cholesky, cholSolve, jacobiEigen, svd, logsumexp, gaussianPdf,
    sigmoid, softmax, relu,
    linearRegression, logisticRegression, knn, kmeans, pca, gmm, svmPegasos, cart, gaussianNB,
    neuralNet, attention, gradCheck
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = SF;
  else global.SF = SF;
})(typeof window !== 'undefined' ? window : globalThis);
