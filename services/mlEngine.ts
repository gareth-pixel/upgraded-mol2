
import { DataRow, RIDGE_FEATURES, TARGET, Tree, RidgeModel, GBDTModel } from '../types';

// --- Math Helpers ---

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

const median = (arr: number[]) => {
  if (arr.length === 0) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 !== 0 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
};

/**
 * Calculates robust coefficients for the baseline using the formula:
 * Ratio = Total Collection / (Collection Days * Total Metric Count)
 */
const calculateRobustBaselineCoeffs = (data: DataRow[]) => {
  const ratiosNotes = data.map(row => {
    const yieldTotal = Number(row[TARGET]) || 0;
    const days = Math.max(1, Number(row['采集天数']) || 1);
    const notesTotal = Math.max(1, Number(row['笔记数']) || 1);
    return yieldTotal / (days * notesTotal);
  });

  const ratiosLikes = data.map(row => {
    const yieldTotal = Number(row[TARGET]) || 0;
    const days = Math.max(1, Number(row['采集天数']) || 1);
    const likesTotal = Math.max(1, Number(row['点赞数']) || 1);
    return yieldTotal / (days * likesTotal);
  });
  
  return [
    Math.max(0, median(ratiosNotes)),
    Math.max(0, median(ratiosLikes))
  ];
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

// --- GBDT Logic ---

const trainRegressionTree = (X: number[][], y: number[], depth: number, maxDepth: number, featureCount: number): Tree => {
  const n = X.length;
  if (depth >= maxDepth || n <= 5) {
    return { featureIndex: -1, threshold: 0, leftValue: y.reduce((a, b) => a + b, 0) / Math.max(1, n) };
  }
  let bestFeature = -1, bestThreshold = 0, minMSE = Infinity;
  for (let f = 0; f < featureCount; f++) {
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
      const mse = leftY.reduce((s, v) => s + Math.pow(v - leftMean, 2), 0) + rightY.reduce((s, v) => s + Math.pow(v - rightMean, 2), 0);
      if (mse < minMSE) { minMSE = mse; bestFeature = f; bestThreshold = threshold; }
    }
  }
  if (bestFeature === -1) return { featureIndex: -1, threshold: 0, leftValue: y.reduce((a, b) => a + b, 0) / Math.max(1, n) };
  const leftX = [], leftY = [], rightX = [], rightY = [];
  for (let i = 0; i < n; i++) {
    if (X[i][bestFeature] <= bestThreshold) { leftX.push(X[i]); leftY.push(y[i]); }
    else { rightX.push(X[i]); rightY.push(y[i]); }
  }
  return {
    featureIndex: bestFeature, threshold: bestThreshold,
    left: trainRegressionTree(leftX, leftY, depth + 1, maxDepth, featureCount),
    right: trainRegressionTree(rightX, rightY, depth + 1, maxDepth, featureCount)
  };
};

const predictTree = (tree: Tree, x: number[]): number => {
  if (tree.featureIndex === -1) return tree.leftValue!;
  return x[tree.featureIndex] <= tree.threshold ? predictTree(tree.left!, x) : predictTree(tree.right!, x);
};

export const trainGBDT = async (data: DataRow[], featureNames: string[], targetName: string, nEstimators = 30, lr = 0.1, maxDepth = 4) => {
  const X = data.map(row => featureNames.map(f => Number(row[f]) || 0));
  const y = data.map(row => Number(row[targetName]) || 0);
  
  const baselineCoeffs = calculateRobustBaselineCoeffs(data);

  const initialMean = y.reduce((a, b) => a + b, 0) / y.length;
  let currentPreds = Array(y.length).fill(initialMean);
  const trees: Tree[] = [];
  for (let i = 0; i < nEstimators; i++) {
    const residuals = y.map((val, idx) => val - currentPreds[idx]);
    const tree = trainRegressionTree(X, residuals, 0, maxDepth, featureNames.length);
    trees.push(tree);
    for (let j = 0; j < y.length; j++) currentPreds[j] += lr * predictTree(tree, X[j]);
  }
  const residualVariance = y.reduce((sum, val, i) => sum + Math.pow(val - currentPreds[i], 2), 0) / Math.max(1, (y.length - 1));
  const residualStd = Math.sqrt(Math.max(0, residualVariance));

  return { initialMean, learningRate: lr, trees, residualStd, featureNames, baselineCoeffs };
};

export interface GuardrailOptions {
  enabled: boolean;
  lowPercent: number; 
  highPercent: number; 
}

export const predictGBDT = (
  model: Omit<GBDTModel, 'type' | 'metrics'>, 
  row: DataRow,
  guardrail: GuardrailOptions = { enabled: true, lowPercent: 30, highPercent: 170 }
) => {
  const x = model.featureNames.map(f => Number(row[f]) || 0);
  
  // 1. GBDT Raw Prediction (predicts 'daily intensity')
  let gbdtDaily = model.initialMean;
  for (const tree of model.trees) {
    gbdtDaily += model.learningRate * predictTree(tree, x);
  }

  let finalDaily = gbdtDaily;
  let isClipped = false;

  // 2. Conditional Guardrail Clipping
  if (guardrail.enabled) {
    const kNotes = model.baselineCoeffs[0];
    const kLikes = model.baselineCoeffs[1];
    
    // Baseline logic: K * Total_Metric
    // Features are now always raw counts (Notes, Likes) for both models.
    const notesTotal = Number(row['笔记数']) || 0;
    const likesTotal = Number(row['点赞数']) || 0;

    const baselineDaily = (kNotes * notesTotal + kLikes * likesTotal) / 2;
    
    const minDaily = baselineDaily * (guardrail.lowPercent / 100);
    const maxDaily = baselineDaily * (guardrail.highPercent / 100);
    
    if (gbdtDaily < minDaily) {
      finalDaily = minDaily;
      isClipped = true;
    } else if (gbdtDaily > maxDaily) {
      finalDaily = maxDaily;
      isClipped = true;
    }
  }

  const margin = 1.28 * model.residualStd;
  return { 
    mean: finalDaily, 
    lowerBound: Math.max(0, finalDaily - margin), 
    upperBound: finalDaily + margin,
    isClipped
  };
};
