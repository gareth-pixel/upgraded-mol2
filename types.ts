
export enum ModelType {
  ONLINE = 'xgboost_model_online',
  RECALL = 'rf_model_recall',
  TELECOM_ONLINE = 'rf_model_telecom',
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
  weights: number[];      
  intercept: number;      
  scaler: {               
    means: number[];
    stds: number[];
  };
  residualStd: number;    
  metrics: TrainingMetrics;
}

export interface Tree {
  featureIndex: number;
  threshold: number;
  leftValue?: number;
  rightValue?: number;
  left?: Tree;
  right?: Tree;
}

export interface GBDTModel {
  type: ModelType;
  initialMean: number;
  learningRate: number;
  trees: Tree[];
  residualStd: number;
  metrics: TrainingMetrics;
  featureNames: string[]; // Added to store which features were used
}

export type ModelData = RidgeModel | GBDTModel;

export const INPUT_FEATURES = [
  '采集天数',
  '笔记数',
  '点赞数',
  '收藏数',
  '评论数'
];

// Ridge models use all features including days
export const RIDGE_FEATURES = [
  '采集天数',
  '日均笔记数',
  '日均点赞数',
  '日均收藏数',
  '日均评论数'
];

// GBDT (Online) specifically excludes '采集天数' from training features
export const GBDT_FEATURES = [
  '日均笔记数',
  '日均点赞数',
  '日均收藏数',
  '日均评论数'
];

export const TARGET = '采集量';
