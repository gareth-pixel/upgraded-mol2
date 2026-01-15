import { ModelType } from './types';

export const MODEL_CONFIGS = {
  [ModelType.ONLINE]: {
    name: '移动在线模型',
    fileName: 'rf_model_online.pkl',
    trainFile: 'train_data_online.xlsx',
    summaryFile: 'formula_info_online.json',
  },
  [ModelType.RECALL]: {
    name: '移动回溯模型',
    fileName: 'rf_model_recall.pkl',
    trainFile: 'train_data_recall.xlsx',
    summaryFile: 'formula_info_recall.json',
  },
  [ModelType.MIX]: {
    name: '移动混合模型',
    fileName: 'rf_model_mix.pkl',
    trainFile: 'train_data_mix.xlsx',
    summaryFile: 'formula_info_mix.json',
  },
};

export const STORAGE_KEYS = {
  DATA: (model: ModelType) => `TRAIN_DATA_${model}`,
  MODEL: (model: ModelType) => `SAVED_MODEL_${model}`,
};