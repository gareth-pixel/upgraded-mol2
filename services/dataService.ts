
import * as XLSX from 'xlsx';
import { DataRow, ModelType, TrainingMetrics, RidgeModel, GBDTModel, ModelData, INPUT_FEATURES, MODEL_FEATURES, TARGET } from '../types';
import { STORAGE_KEYS, MODEL_CONFIGS } from '../constants';
import { trainRidge, predictRidge, trainGBDT, predictGBDT, calculateR2, calculateMAE } from './mlEngine';
import { dbService } from './db';

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
  
  let content: any = {
    "模型类型": modelType === ModelType.ONLINE ? "梯度提升回归树 (XGBoost)" : "岭回归 (Ridge Regression)",
    "子模型": MODEL_CONFIGS[modelType].name,
    "样本量": model.metrics.sampleSize,
    "R²": model.metrics.r2.toFixed(4),
    "MAE": model.metrics.mae.toFixed(4),
    "更新时间": model.metrics.lastUpdated
  };

  if (modelType !== ModelType.ONLINE) {
    const ridge = model as RidgeModel;
    content["权重分配"] = MODEL_FEATURES.reduce((acc, f, i) => ({ ...acc, [f]: ridge.weights[i].toFixed(4) }), {});
    content["截距"] = ridge.intercept.toFixed(4);
  } else {
    content["树数量"] = (model as GBDTModel).trees.length;
    content["学习率"] = (model as GBDTModel).learningRate;
  }
  
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

  onProgress?.("正在进行特征派生...");
  const processedData = data.map(preprocessRow);

  onProgress?.(`正在执行模型训练 (样本量: ${data.length})...`);
  await new Promise(resolve => setTimeout(resolve, 100));
  
  let modelPayload: ModelData;
  let predictions: number[];
  const yTrue = processedData.map(r => Number(r[TARGET]));

  if (modelType === ModelType.ONLINE) {
    const params = await trainGBDT(processedData);
    predictions = processedData.map(r => predictGBDT(params, r).mean);
    modelPayload = { type: modelType, ...params, metrics: { r2: 0, mae: 0, sampleSize: 0, lastUpdated: '' } };
  } else {
    const params = await trainRidge(processedData);
    predictions = processedData.map(r => predictRidge(params, r).mean);
    modelPayload = { type: modelType, ...params, metrics: { r2: 0, mae: 0, sampleSize: 0, lastUpdated: '' } };
  }
  
  const metrics: TrainingMetrics = {
    r2: calculateR2(yTrue, predictions),
    mae: calculateMAE(yTrue, predictions),
    sampleSize: data.length,
    lastUpdated: new Date().toLocaleString()
  };

  modelPayload.metrics = metrics;

  onProgress?.("同步到本地数据库...");
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
  onProgress: (msg: string) => void
): Promise<DataRow[]> => {
  onProgress("正在加载预测模型...");
  const model = await dbService.getModel(STORAGE_KEYS.MODEL(modelType)) as ModelData;
  if (!model) throw new Error("该模型尚未训练。");

  onProgress("处理预测数据...");
  const data = await readFile(file);
  if (data.length === 0) throw new Error("文件为空");
  const error = validateColumns(data[0], false); 
  if (error) throw new Error(error);

  const results = data.map(row => {
    const processedRow = preprocessRow(row);
    const preds = modelType === ModelType.ONLINE 
      ? predictGBDT(model as GBDTModel, processedRow)
      : predictRidge(model as RidgeModel, processedRow);
    return {
      ...row,
      '预测采集量': Math.round(preds.mean),
      '预测下限': Math.round(preds.lowerBound),
      '预测上限': Math.round(preds.upperBound)
    };
  });

  return results;
};

export const exportPredictionResults = (results: DataRow[], originalFileName: string, modelType: ModelType) => {
  const ws = XLSX.utils.json_to_sheet(results);
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, "Predictions");
  const modelSuffix = modelType.split('_')[0];
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
