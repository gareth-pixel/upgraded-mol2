
export enum ModelType {
  ONLINE = 'rf_model_online',
  RECALL = 'rf_model_recall',
  MIX = 'rf_model_mix',
}

export interface DataRow {
  [key: string]: any;
}

export interface TrainingMetrics {
  r2: number;
  mae: number;
  sampleSize: number;
  lastUpdated: string;
}

export interface PredictionResult {
  mean: number;
  lowerBound: number; // 10%
  upperBound: number; // 90%
}

export interface RandomForestModel {
  type: ModelType;
  trees: DecisionTreeNode[];
  metrics: TrainingMetrics;
}

// Minimal structure for a decision tree node
export interface DecisionTreeNode {
  isLeaf: boolean;
  value?: number; // For leaf
  feature?: string;
  threshold?: number;
  left?: DecisionTreeNode;
  right?: DecisionTreeNode;
}

// Features required in the input file (Raw Totals)
export const INPUT_FEATURES = [
  '采集天数',
  '笔记数',
  '点赞数',
  '收藏数',
  '评论数'
];

// Features actually used by the Random Forest model
export const MODEL_FEATURES = [
  '采集天数',
  '日均笔记数',
  '日均点赞数',
  '日均收藏数',
  '日均评论数'
];

export const TARGET = '采集量';
