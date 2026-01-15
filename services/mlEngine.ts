
import { DataRow, MODEL_FEATURES, TARGET } from '../types';

// --- Linear Algebra Helpers ---

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

// --- Ridge Regression Logic ---

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

/**
 * Standardize features: (x - mean) / std
 */
const getScaler = (X: number[][]) => {
  const n = X.length;
  const p = X[0].length;
  const means = Array(p).fill(0);
  const stds = Array(p).fill(0);

  for (let j = 0; j < p; j++) {
    const col = X.map(row => row[j]);
    means[j] = col.reduce((a, b) => a + b, 0) / n;
    const variance = col.reduce((a, b) => a + Math.pow(b - means[j], 2), 0) / n;
    stds[j] = Math.sqrt(variance) || 1; // Prevent division by zero
  }
  return { means, stds };
};

const scale = (X: number[][], scaler: { means: number[], stds: number[] }) => {
  return X.map(row => row.map((val, j) => (val - scaler.means[j]) / scaler.stds[j]));
};

/**
 * Train Ridge Regression
 * w = (X^T X + alpha*I)^-1 X^T y
 */
export const trainRidge = async (data: DataRow[], alpha = 1.0) => {
  const n = data.length;
  const p = MODEL_FEATURES.length;

  const X_raw = data.map(row => MODEL_FEATURES.map(f => Number(row[f]) || 0));
  const y = data.map(row => [Number(row[TARGET]) || 0]);

  // 1. Scale Features
  const scaler = getScaler(X_raw);
  const X_scaled = scale(X_raw, scaler);

  // 2. Add Bias term (column of 1s)
  const X = X_scaled.map(row => [1, ...row]);

  // 3. Normal Equation with L2
  const XT = transpose(X);
  const XTX = multiply(XT, X);
  
  // Apply alpha to identity (don't regularize intercept at index 0)
  for (let i = 1; i < XTX.length; i++) {
    XTX[i][i] += alpha;
  }

  const XTX_inv = invert(XTX);
  const XTy = multiply(XT, y);
  const weights_matrix = multiply(XTX_inv, XTy);

  const weights_all = weights_matrix.map(row => row[0]);
  const intercept = weights_all[0];
  const coefficients = weights_all.slice(1);

  // Calculate residuals for confidence intervals
  const yTrue = y.map(row => row[0]);
  const yPred = X.map(row => {
    let sum = weights_all[0];
    for (let i = 1; i < weights_all.length; i++) {
        sum += weights_all[i] * row[i];
    }
    return sum;
  });

  const residualVariance = yTrue.reduce((sum, val, i) => sum + Math.pow(val - yPred[i], 2), 0) / (n - p - 1);
  const residualStd = Math.sqrt(Math.max(0, residualVariance));

  return {
    weights: coefficients,
    intercept,
    scaler,
    residualStd
  };
};

export const predictRidge = (model: { weights: number[], intercept: number, scaler: { means: number[], stds: number[] }, residualStd: number }, row: DataRow) => {
  const x_raw = MODEL_FEATURES.map(f => Number(row[f]) || 0);
  const x_scaled = x_raw.map((val, i) => (val - model.scaler.means[i]) / model.scaler.stds[i]);
  
  let mean = model.intercept;
  for (let i = 0; i < x_scaled.length; i++) {
    mean += x_scaled[i] * model.weights[i];
  }

  // Use 1.28 for 80% confidence interval (Z-score for 10th and 90th percentile)
  const margin = 1.28 * model.residualStd;
  
  return {
    mean,
    lowerBound: Math.max(0, mean - margin),
    upperBound: mean + margin
  };
};
