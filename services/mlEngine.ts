
import { DataRow, MODEL_FEATURES, TARGET, Tree, RidgeModel, GBDTModel } from '../types';

// --- Linear Algebra Helpers for Ridge ---

const transpose = (m: number[][]) => m[0].map((_, i) => m.map(row => row[i]));

const multiply = (a: number[][], b: number[][]) => {
  const result = Array(a.length).fill(0).map(() => Array(b[0].length).fill(0));
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b[0].length; j++) {
      for (let k = 0; k < b.length; k++) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return result;
};

const invert = (m: number[][]) => {
  const n = m.length;
  const identity = Array(n).fill(0).map((_, i) => Array(n).fill(0).map((_, j) => (i === j ? 1 : 0)));
  const copy = m.map(row => [...row]);

  for (let i = 0; i < n; i++) {
    let pivot = copy[i][i];
    if (Math.abs(pivot) < 1e-10) pivot = 1e-10; 
    
    for (let j = 0; j < n; j++) {
      copy[i][j] /= pivot;
      identity[i][j] /= pivot;
    }
    for (let k = 0; k < n; k++) {
      if (k !== i) {
        const factor = copy[k][i];
        for (let j = 0; j < n; j++) {
          copy[k][j] -= factor * copy[i][j];
          identity[k][j] -= factor * identity[i][j];
        }
      }
    }
  }
  return identity;
};

// --- Evaluation Metrics ---

export const calculateMAE = (yTrue: number[], yPred: number[]): number => {
  if (yTrue.length === 0) return 0;
  return yTrue.reduce((sum, val, i) => sum + Math.abs(val - yPred[i]), 0) / yTrue.length;
};

export const calculateR2 = (yTrue: number[], yPred: number[]): number => {
  if (yTrue.length === 0) return 0;
  const meanY = yTrue.reduce((a, b) => a + b, 0) / yTrue.length;
  const ssTot = yTrue.reduce((sum, val) => sum + Math.pow(val - meanY, 2), 0);
  const ssRes = yTrue.reduce((sum, val, i) => sum + Math.pow(val - yPred[i], 2), 0);
  return ssTot === 0 ? 0 : 1 - (ssRes / ssTot);
};

// --- Ridge Regression Logic ---

const getScaler = (X: number[][]) => {
  const n = X.length;
  const p = X[0].length;
  const means = Array(p).fill(0);
  const stds = Array(p).fill(0);

  for (let j = 0; j < p; j++) {
    const col = X.map(row => row[j]);
    means[j] = col.reduce((a, b) => a + b, 0) / n;
    const variance = col.reduce((a, b) => a + Math.pow(b - means[j], 2), 0) / n;
    stds[j] = Math.sqrt(variance) || 1; 
  }
  return { means, stds };
};

const scale = (X: number[][], scaler: { means: number[], stds: number[] }) => {
  return X.map(row => row.map((val, j) => (val - scaler.means[j]) / scaler.stds[j]));
};

export const trainRidge = async (data: DataRow[], alpha = 1.0) => {
  const n = data.length;
  const p = MODEL_FEATURES.length;
  const X_raw = data.map(row => MODEL_FEATURES.map(f => Number(row[f]) || 0));
  const y = data.map(row => [Number(row[TARGET]) || 0]);
  const scaler = getScaler(X_raw);
  const X_scaled = scale(X_raw, scaler);
  const X = X_scaled.map(row => [1, ...row]);
  const XT = transpose(X);
  const XTX = multiply(XT, X);
  for (let i = 1; i < XTX.length; i++) XTX[i][i] += alpha;
  const XTX_inv = invert(XTX);
  const XTy = multiply(XT, y);
  const weights_matrix = multiply(XTX_inv, XTy);
  const weights_all = weights_matrix.map(row => row[0]);
  const yTrue = y.map(row => row[0]);
  const yPred = X.map(row => {
    let sum = weights_all[0];
    for (let i = 1; i < weights_all.length; i++) sum += weights_all[i] * row[i];
    return sum;
  });
  const residualVariance = yTrue.reduce((sum, val, i) => sum + Math.pow(val - yPred[i], 2), 0) / (n - p - 1);
  const residualStd = Math.sqrt(Math.max(0, residualVariance));
  return { weights: weights_all.slice(1), intercept: weights_all[0], scaler, residualStd };
};

export const predictRidge = (model: Omit<RidgeModel, 'type' | 'metrics'>, row: DataRow) => {
  const x_raw = MODEL_FEATURES.map(f => Number(row[f]) || 0);
  const x_scaled = x_raw.map((val, i) => (val - model.scaler.means[i]) / model.scaler.stds[i]);
  let mean = model.intercept;
  for (let i = 0; i < x_scaled.length; i++) mean += x_scaled[i] * model.weights[i];
  const margin = 1.28 * model.residualStd;
  return { mean, lowerBound: Math.max(0, mean - margin), upperBound: mean + margin };
};

// --- GBDT (XGBoost-like) Logic ---

const trainRegressionTree = (X: number[][], y: number[], depth: number, maxDepth: number): Tree => {
  const n = X.length;
  if (depth >= maxDepth || n <= 5) {
    return { featureIndex: -1, threshold: 0, leftValue: y.reduce((a, b) => a + b, 0) / n };
  }

  let bestFeature = -1;
  let bestThreshold = 0;
  let minMSE = Infinity;

  for (let f = 0; f < MODEL_FEATURES.length; f++) {
    const values = X.map(row => row[f]).sort((a, b) => a - b);
    const uniqueValues = Array.from(new Set(values));
    
    for (let i = 0; i < uniqueValues.length - 1; i++) {
      const threshold = (uniqueValues[i] + uniqueValues[i+1]) / 2;
      const leftY = [], rightY = [];
      for (let j = 0; j < n; j++) {
        if (X[j][f] <= threshold) leftY.push(y[j]);
        else rightY.push(y[j]);
      }
      if (leftY.length === 0 || rightY.length === 0) continue;
      
      const leftMean = leftY.reduce((a, b) => a + b, 0) / leftY.length;
      const rightMean = rightY.reduce((a, b) => a + b, 0) / rightY.length;
      const mse = leftY.reduce((s, v) => s + Math.pow(v - leftMean, 2), 0) + 
                  rightY.reduce((s, v) => s + Math.pow(v - rightMean, 2), 0);
      
      if (mse < minMSE) {
        minMSE = mse;
        bestFeature = f;
        bestThreshold = threshold;
      }
    }
  }

  if (bestFeature === -1) return { featureIndex: -1, threshold: 0, leftValue: y.reduce((a, b) => a + b, 0) / n };

  const leftX = [], leftY = [], rightX = [], rightY = [];
  for (let i = 0; i < n; i++) {
    if (X[i][bestFeature] <= bestThreshold) {
      leftX.push(X[i]); leftY.push(y[i]);
    } else {
      rightX.push(X[i]); rightY.push(y[i]);
    }
  }

  return {
    featureIndex: bestFeature,
    threshold: bestThreshold,
    left: trainRegressionTree(leftX, leftY, depth + 1, maxDepth),
    right: trainRegressionTree(rightX, rightY, depth + 1, maxDepth)
  };
};

const predictTree = (tree: Tree, x: number[]): number => {
  if (tree.featureIndex === -1) return tree.leftValue!;
  if (x[tree.featureIndex] <= tree.threshold) return predictTree(tree.left!, x);
  return predictTree(tree.right!, x);
};

export const trainGBDT = async (data: DataRow[], nEstimators = 30, lr = 0.1, maxDepth = 4) => {
  const X = data.map(row => MODEL_FEATURES.map(f => Number(row[f]) || 0));
  const y = data.map(row => Number(row[TARGET]) || 0);
  const initialMean = y.reduce((a, b) => a + b, 0) / y.length;
  let currentPreds = Array(y.length).fill(initialMean);
  const trees: Tree[] = [];

  for (let i = 0; i < nEstimators; i++) {
    const residuals = y.map((val, idx) => val - currentPreds[idx]);
    const tree = trainRegressionTree(X, residuals, 0, maxDepth);
    trees.push(tree);
    for (let j = 0; j < y.length; j++) {
      currentPreds[j] += lr * predictTree(tree, X[j]);
    }
  }

  const residualVariance = y.reduce((sum, val, i) => sum + Math.pow(val - currentPreds[i], 2), 0) / (y.length - 1);
  const residualStd = Math.sqrt(Math.max(0, residualVariance));

  return { initialMean, learningRate: lr, trees, residualStd };
};

export const predictGBDT = (model: Omit<GBDTModel, 'type' | 'metrics'>, row: DataRow) => {
  const x = MODEL_FEATURES.map(f => Number(row[f]) || 0);
  let mean = model.initialMean;
  for (const tree of model.trees) {
    mean += model.learningRate * predictTree(tree, x);
  }
  const margin = 1.28 * model.residualStd;
  return { mean, lowerBound: Math.max(0, mean - margin), upperBound: mean + margin };
};
