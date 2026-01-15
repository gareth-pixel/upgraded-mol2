
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

export interface RidgeModel {
  type: ModelType;
  weights: number[];      // Coefficients for MODEL_FEATURES
  intercept: number;      // Bias term
  scaler: {               // Mean and Std for normalization
    means: number[];
    stds: number[];
  };
  residualStd: number;    // Used for confidence interval
  metrics: TrainingMetrics;
}

// Features required in the input file (Raw Totals)
export const INPUT_FEATURES = [
  '采集天数',
  '笔记数',
  '点赞数',
  '收藏数',
  '评论数'
];

// Features actually used by the model
export const MODEL_FEATURES = [
  '采集天数',
  '日均笔记数',
  '日均点赞数',
  '日均收藏数',
  '日均评论数'
];

export const TARGET = '采集量';
