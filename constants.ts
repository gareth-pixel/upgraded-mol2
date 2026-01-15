
import { ModelType } from './types';

export const MODEL_CONFIGS = {
  [ModelType.ONLINE]: {
    name: '移动在线模型 (XGBoost)',
    fileName: 'xgboost_model_online.json',
    trainFile: 'train_data_online.xlsx',
    summaryFile: 'formula_info_online.json',
  },
  [ModelType.RECALL]: {
    name: '移动回溯模型 (Ridge)',
    fileName: 'rf_model_recall.json',
    trainFile: 'train_data_recall.xlsx',
    summaryFile: 'formula_info_recall.json',
  },
  [ModelType.TELECOM_ONLINE]: {
    name: '电信在线模型 (Ridge)',
    fileName: 'rf_model_telecom.json',
    trainFile: 'train_data_telecom.xlsx',
    summaryFile: 'formula_info_telecom.json',
  },
};

export const STORAGE_KEYS = {
  DATA: (model: ModelType) => `TRAIN_DATA_${model}`,
  MODEL: (model: ModelType) => `SAVED_MODEL_${model}`,
};
