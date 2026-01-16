
import * as XLSX from 'xlsx';
import { DataRow, ModelType, TrainingMetrics, RidgeModel, GBDTModel, ModelData, INPUT_FEATURES, RIDGE_FEATURES, GBDT_FEATURES, TARGET } from '../types';
import { STORAGE_KEYS, MODEL_CONFIGS } from '../constants';
// Removed trainRidge and predictRidge from mlEngine imports as they are not exported and the engine now uses GBDT exclusively.
import { trainGBDT, predictGBDT, calculateR2, calculateMAE, GuardrailOptions } from './mlEngine';
import { dbService } from './db';

/**
 * Preprocessing now only adds derived daily features for legacy or optional use.
 * The core RECALL model will now use raw GBDT_FEATURES for better scaling.
 */
export const preprocessRow = (row: DataRow): DataRow => {
  const days = Math.max(0, Number(row['采集天数']) || 0);
  const processed: DataRow = { ...row };
  processed['日均笔记数'] = days > 0 ? (Number(row['笔记数']) || 0) / days : 0;
  processed['日均点赞数'] = days > 0 ? (Number(row['点赞数']) || 0) / days : 0;
  processed['日均收藏数'] = days > 0 ? (Number(row['收藏数']) || 0) / days : 0;
  processed['日均评论数'] = days > 0 ? (Number(row['评论数']) || 0) / days : 0;
  return processed;
};

export const validateColumns = (row: any, isTraining: boolean): string | null => {
  const missing = [];
  for (const f of INPUT_FEATURES) {
    if (row[f] === undefined || row[f] === null || row[f] === '') missing.push(f);
  }
  if (isTraining && (row[TARGET] === undefined || row[TARGET] === null)) {
    missing.push(TARGET);
  }
  if (missing.length > 0) return `缺失列: ${missing.join(', ')}`;
  return null;
};

export const getStoredMetrics = async (modelType: ModelType): Promise<TrainingMetrics | null> => {
  const modelData = await dbService.getModel(STORAGE_KEYS.MODEL(modelType)) as ModelData;
  return modelData ? modelData.metrics : null;
};

export const clearModelData = async (modelType: ModelType) => {
  await dbService.saveModel(STORAGE_KEYS.MODEL(modelType), null);
  await dbService.saveData(STORAGE_KEYS.DATA(modelType), []);
};

export const downloadTrainingData = async (modelType: ModelType) => {
  const data: DataRow[] = await dbService.getData(STORAGE_KEYS.DATA(modelType));
  if (!data || data.length === 0) throw new Error("当前模型暂无累积训练数据");
  const ws = XLSX.utils.json_to_sheet(data);
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, "TrainingData");
  XLSX.writeFile(wb, MODEL_CONFIGS[modelType].trainFile);
};

export const generateTrainTemplate = () => {
  const ws = XLSX.utils.aoa_to_sheet([[...INPUT_FEATURES, TARGET]]);
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, "Training_Template");
  XLSX.writeFile(wb, "train_template.xlsx");
};

export const generatePredictionTemplate = () => {
  const ws = XLSX.utils.aoa_to_sheet([[...INPUT_FEATURES]]);
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, "Prediction_Template");
  XLSX.writeFile(wb, "predict_template.xlsx");
};

export const downloadSummary = async (modelType: ModelType) => {
  const model = await dbService.getModel(STORAGE_KEYS.MODEL(modelType)) as ModelData;
  if (!model) throw new Error("该模型暂无训练数据");
  
  const isRecall = modelType === ModelType.RECALL;

  let content: any = {
    "模型类型": "梯度提升回归树 (XGBoost)",
    "子模型": MODEL_CONFIGS[modelType].name,
    "样本量": model.metrics.sampleSize,
    "R²": model.metrics.r2.toFixed(4),
    "MAE": model.metrics.mae.toFixed(4),
    "更新时间": model.metrics.lastUpdated
  };

  const gbdt = model as GBDTModel;
  content["算法特征"] = gbdt.featureNames;
  
  if (isRecall) {
    content["稳健基线系数"] = {
      "业务产出效率 (k_notes)": gbdt.baselineCoeffs[0].toFixed(6),
      "业务产出效率 (k_likes)": gbdt.baselineCoeffs[1].toFixed(6)
    };
    content["预测逻辑"] = "强度学习: 基于指标总量预测日均产量 + 天数线性放大";
    content["限幅说明"] = "支持基于指标总量的日均基线限幅。";
  } else {
    content["预测逻辑"] = "单日总量直接推理 (Yield/1Day)";
    content["限幅说明"] = "不适用 (采用纯模型原始输出)";
  }
  
  content["树数量"] = gbdt.trees.length;
  content["学习率"] = gbdt.learningRate;
  
  const blob = new Blob([JSON.stringify(content, null, 2)], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = MODEL_CONFIGS[modelType].summaryFile.replace('.json', '.txt');
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

export const trainFromData = async (
  data: DataRow[],
  modelType: ModelType,
  onProgress?: (msg: string) => void
): Promise<TrainingMetrics> => {
  if (data.length === 0) throw new Error("训练数据为空");

  const isRecall = modelType === ModelType.RECALL;
  
  onProgress?.("正在进行特征预处理...");
  const processedData = data.map(preprocessRow);

  onProgress?.(`正在执行模型训练 (${isRecall ? '产量强度学习' : '总量直接推理'})...`);
  await new Promise(resolve => setTimeout(resolve, 100));
  
  // Normalized target for RECALL, Raw target for ONLINE
  const trainingDataWithTarget = processedData.map(r => {
    const days = isRecall ? Math.max(1, Number(r['采集天数']) || 1) : 1;
    return {
      ...r,
      'TRAIN_TARGET': (Number(r[TARGET]) || 0) / days
    };
  });

  /**
   * IMPORTANT: Both models now use GBDT_FEATURES (raw counts).
   * RECALL learns: Yield/Days ~ f(Raw_Metrics)
   * ONLINE learns: Yield ~ f(Raw_Metrics)
   */
  const featuresToUse = GBDT_FEATURES;
  const params = await trainGBDT(trainingDataWithTarget, featuresToUse, 'TRAIN_TARGET');
  
  const predictions = processedData.map(r => {
    const basePred = predictGBDT(params, r, { enabled: false, lowPercent: 0, highPercent: 0 }).mean;
    const days = isRecall ? (Number(r['采集天数']) || 0) : 1;
    return basePred * days; 
  });

  const metrics: TrainingMetrics = {
    r2: calculateR2(processedData.map(r => Number(r[TARGET])), predictions),
    mae: calculateMAE(processedData.map(r => Number(r[TARGET])), predictions),
    sampleSize: data.length,
    lastUpdated: new Date().toLocaleString()
  };

  const modelPayload: GBDTModel = { 
    type: modelType, 
    ...params, 
    metrics 
  };

  onProgress?.("同步本地数据库...");
  await dbService.saveData(STORAGE_KEYS.DATA(modelType), data);
  await dbService.saveModel(STORAGE_KEYS.MODEL(modelType), modelPayload);

  return metrics;
};

export const handleTrain = async (
  file: File, 
  modelType: ModelType,
  onProgress: (msg: string) => void
): Promise<TrainingMetrics> => {
  onProgress("读取文件...");
  const data = await readFile(file);
  if (data.length === 0) throw new Error("文件为空");
  const error = validateColumns(data[0], true);
  if (error) throw new Error(error);

  onProgress("合并历史数据...");
  const oldData: DataRow[] = (await dbService.getData(STORAGE_KEYS.DATA(modelType))) || [];
  return await trainFromData([...oldData, ...data], modelType, onProgress);
};

export const handlePredict = async (
  file: File,
  modelType: ModelType,
  onProgress: (msg: string) => void,
  guardrailOptions?: GuardrailOptions
): Promise<DataRow[]> => {
  onProgress("正在加载预测模型...");
  const model = await dbService.getModel(STORAGE_KEYS.MODEL(modelType)) as GBDTModel;
  if (!model) throw new Error("该模型尚未训练。");

  onProgress("执行预测计算...");
  const data = await readFile(file);
  if (data.length === 0) throw new Error("文件为空");
  const error = validateColumns(data[0], false); 
  if (error) throw new Error(error);

  const isRecall = modelType === ModelType.RECALL;

  const results = data.map(row => {
    // RECALL now uses raw indicators to predict daily intensity
    const processedRow = preprocessRow(row);
    
    // Force guardrail off for non-recall models
    const activeGuardrail = isRecall 
      ? guardrailOptions 
      : { enabled: false, lowPercent: 30, highPercent: 170 };

    const preds = predictGBDT(model, processedRow, activeGuardrail);
    
    /**
     * Scaling Logic:
     * preds.mean is always 'yield per day' (either directly or normalized during training).
     * For RECALL: Multiply by input days.
     * For ONLINE: multiplier is 1.
     */
    const multiplier = isRecall ? (Number(row['采集天数']) || 0) : 1;
    const mean = preds.mean * multiplier;
    const lower = preds.lowerBound * multiplier;
    const upper = preds.upperBound * multiplier;

    return {
      ...row,
      '预测采集量': Math.round(mean),
      '预测下限': Math.round(lower),
      '预测上限': Math.round(upper),
      '_IS_CLIPPED': preds.isClipped
    };
  });

  return results;
};

export const exportPredictionResults = (results: DataRow[], originalFileName: string, modelType: ModelType) => {
  const ws = XLSX.utils.json_to_sheet(results.map(({ _IS_CLIPPED, ...rest }) => rest));
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, "Predictions");
  const modelSuffix = modelType.split('_')[1] || 'model';
  XLSX.writeFile(wb, `${originalFileName.split('.')[0]}_${modelSuffix}_predicted.xlsx`);
};

export const getModelExportData = async (modelType: ModelType) => {
  const modelData = await dbService.getModel(STORAGE_KEYS.MODEL(modelType));
  const dataset = await dbService.getData(STORAGE_KEYS.DATA(modelType));
  if (!modelData && (!dataset || dataset.length === 0)) return null;
  return { [modelType]: { model: modelData, data: dataset || [] } };
};

export const restoreModelFromRemote = async (
  modelType: ModelType, 
  remoteData: any,
  onProgress: (msg: string) => void
): Promise<TrainingMetrics | null> => {
  const targetData = remoteData[modelType];
  if (!targetData) return null;

  onProgress("正在恢复云端数据...");
  const data: DataRow[] = targetData.data || [];
  const model: ModelData | null = targetData.model;

  if (data.length > 0) await dbService.saveData(STORAGE_KEYS.DATA(modelType), data);
  if (model) {
    await dbService.saveModel(STORAGE_KEYS.MODEL(modelType), model);
    return model.metrics;
  } else if (data.length > 0) {
    return await trainFromData(data, modelType, onProgress);
  }
  return null;
};

const readFile = (file: File): Promise<DataRow[]> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const wb = XLSX.read(e.target?.result, { type: 'binary' });
        resolve(XLSX.utils.sheet_to_json(wb.Sheets[wb.SheetNames[0]]) as DataRow[]);
      } catch (err) { reject(err); }
    };
    reader.readAsBinaryString(file);
  });
};
