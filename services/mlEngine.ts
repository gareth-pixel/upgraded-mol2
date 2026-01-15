
import { DataRow, DecisionTreeNode, MODEL_FEATURES, TARGET } from '../types';

// Constants for RF
const N_ESTIMATORS = 200;
const MIN_SAMPLES_SPLIT = 5;
const MAX_DEPTH = 15;
const MAX_FEATURES_RATIO = 0.7; // Use 70% of features for split consideration

/**
 * Calculates Mean Absolute Error
 */
export const calculateMAE = (yTrue: number[], yPred: number[]): number => {
  if (yTrue.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < yTrue.length; i++) {
    sum += Math.abs(yTrue[i] - yPred[i]);
  }
  return sum / yTrue.length;
};

/**
 * Calculates R-squared
 */
export const calculateR2 = (yTrue: number[], yPred: number[]): number => {
  if (yTrue.length === 0) return 0;
  const meanY = yTrue.reduce((a, b) => a + b, 0) / yTrue.length;
  let ssTot = 0;
  let ssRes = 0;
  for (let i = 0; i < yTrue.length; i++) {
    ssTot += Math.pow(yTrue[i] - meanY, 2);
    ssRes += Math.pow(yTrue[i] - yPred[i], 2);
  }
  if (ssTot === 0) return 0;
  return 1 - (ssRes / ssTot);
};

// --- Random Forest Implementation ---

/**
 * Bootstrapping: Sample with replacement
 */
const bootstrapSample = (data: DataRow[]): DataRow[] => {
  const n = data.length;
  const sample: DataRow[] = [];
  for (let i = 0; i < n; i++) {
    const idx = Math.floor(Math.random() * n);
    sample.push(data[idx]);
  }
  return sample;
};

/**
 * Build a Decision Tree
 */
const buildTree = (data: DataRow[], depth: number): DecisionTreeNode => {
  const yValues = data.map(r => Number(r[TARGET]));
  const meanVal = yValues.reduce((a, b) => a + b, 0) / yValues.length;

  // Stopping criteria
  if (depth >= MAX_DEPTH || data.length < MIN_SAMPLES_SPLIT || new Set(yValues).size === 1) {
    return { isLeaf: true, value: meanVal };
  }

  let bestSplit = { feature: '', threshold: 0, varianceReduction: -Infinity, left: [] as DataRow[], right: [] as DataRow[] };
  
  // Random subset of features - Using MODEL_FEATURES
  const features = [...MODEL_FEATURES].sort(() => 0.5 - Math.random()).slice(0, Math.ceil(MODEL_FEATURES.length * MAX_FEATURES_RATIO));

  const currentVariance = calculateVariance(yValues);

  for (const feature of features) {
    // Get unique values to test as thresholds
    const values = Array.from(new Set(data.map(d => Number(d[feature]))));
    // Optimization: Don't test every single value if there are too many, just sample some
    const testValues = values.length > 20 ? values.sort(() => 0.5 - Math.random()).slice(0, 20) : values;

    for (const threshold of testValues) {
      const left = [];
      const right = [];
      for (const row of data) {
        if (Number(row[feature]) <= threshold) left.push(row);
        else right.push(row);
      }

      if (left.length === 0 || right.length === 0) continue;

      const varLeft = calculateVariance(left.map(r => Number(r[TARGET])));
      const varRight = calculateVariance(right.map(r => Number(r[TARGET])));
      
      const reduction = currentVariance - ((left.length / data.length) * varLeft + (right.length / data.length) * varRight);

      if (reduction > bestSplit.varianceReduction) {
        bestSplit = { feature, threshold, varianceReduction: reduction, left, right };
      }
    }
  }

  if (bestSplit.varianceReduction === -Infinity) {
    return { isLeaf: true, value: meanVal };
  }

  return {
    isLeaf: false,
    feature: bestSplit.feature,
    threshold: bestSplit.threshold,
    left: buildTree(bestSplit.left, depth + 1),
    right: buildTree(bestSplit.right, depth + 1)
  };
};

const calculateVariance = (values: number[]): number => {
  if (values.length === 0) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
};

const predictTree = (node: DecisionTreeNode, row: DataRow): number => {
  if (node.isLeaf) return node.value!;
  const val = Number(row[node.feature!]);
  if (val <= node.threshold!) {
    return predictTree(node.left!, row);
  } else {
    return predictTree(node.right!, row);
  }
};

/**
 * Main Training Function (Async to not freeze UI)
 */
export const trainRandomForest = async (data: DataRow[]): Promise<{ trees: DecisionTreeNode[] }> => {
  const trees: DecisionTreeNode[] = [];
  
  // Since we are in browser, we chunk the work
  const chunkSize = 10;
  
  for (let i = 0; i < N_ESTIMATORS; i += chunkSize) {
    // Allow UI to breathe
    await new Promise(resolve => setTimeout(resolve, 0)); 
    
    for (let j = 0; j < chunkSize && (i + j) < N_ESTIMATORS; j++) {
      const sample = bootstrapSample(data);
      trees.push(buildTree(sample, 0));
    }
  }
  
  return { trees };
};

export const predictForest = (trees: DecisionTreeNode[], row: DataRow) => {
  const predictions = trees.map(tree => predictTree(tree, row));
  
  // Sort for quantiles
  predictions.sort((a, b) => a - b);
  
  const n = predictions.length;
  const mean = predictions.reduce((a, b) => a + b, 0) / n;
  
  // 10th and 90th percentile
  const idx10 = Math.floor(n * 0.1);
  const idx90 = Math.floor(n * 0.9);
  
  // Guard against small tree counts
  const lowerBound = predictions[Math.max(0, idx10)];
  const upperBound = predictions[Math.min(n - 1, idx90)];

  return { mean, lowerBound, upperBound };
};
