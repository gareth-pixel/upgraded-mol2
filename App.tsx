
import React, { useState, useEffect, useRef } from 'react';
import { Download, Upload, BarChart2, FileSpreadsheet, Activity, AlertCircle, CheckCircle, X, Save, Trash2, Database, Github, Settings, CloudUpload, CloudDownload } from 'lucide-react';
import { ModelType, TrainingMetrics, DataRow, INPUT_FEATURES } from './types';
import { MODEL_CONFIGS } from './constants';
import { getStoredMetrics, generateTrainTemplate, generatePredictionTemplate, downloadSummary, handleTrain, handlePredict, exportPredictionResults, clearModelData, downloadTrainingData, getModelExportData, restoreModelFromRemote } from './services/dataService';
import { getGitHubConfig, saveGitHubConfig, uploadToGitHub, fetchFromGitHub, GitHubConfig } from './services/github';
import { Button } from './components/Button';
import { Card } from './components/Card';

const App: React.FC = () => {
  const [currentModel, setCurrentModel] = useState<ModelType>(ModelType.ONLINE);
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [status, setStatus] = useState<{ type: 'idle' | 'loading' | 'success' | 'error', msg: string }>({ type: 'idle', msg: '' });
  const [isInitializing, setIsInitializing] = useState(true);
  
  // Prediction Preview State
  const [previewData, setPreviewData] = useState<DataRow[] | null>(null);
  const [predictFileName, setPredictFileName] = useState<string>("");

  // GitHub Settings State
  const [showSettings, setShowSettings] = useState(false);
  const [ghConfig, setGhConfig] = useState<GitHubConfig>({ token: '', owner: '', repo: '', path: 'public/data/model_result.json' });

  // File inputs refs
  const trainInputRef = useRef<HTMLInputElement>(null);
  const predictInputRef = useRef<HTMLInputElement>(null);

  // Load metrics on model switch and clear preview
  useEffect(() => {
    const loadMetrics = async () => {
      setIsInitializing(true);
      setPreviewData(null);
      setPredictFileName("");
      setStatus({ type: 'idle', msg: '' });
      try {
        const m = await getStoredMetrics(currentModel);
        setMetrics(m);
      } catch (e) {
        console.error("Failed to load metrics", e);
      } finally {
        setIsInitializing(false);
      }
    };
    loadMetrics();
  }, [currentModel]);

  // Load Settings on Mount
  useEffect(() => {
    const saved = getGitHubConfig();
    if (saved) setGhConfig(saved);
  }, []);

  const onTrainFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    
    try {
      setStatus({ type: 'loading', msg: '初始化训练...' });
      const newMetrics = await handleTrain(file, currentModel, (msg) => setStatus({ type: 'loading', msg }));
      setMetrics(newMetrics);
      setStatus({ type: 'success', msg: '训练完成！特征派生已自动处理，模型已保存。' });
      setPreviewData(null); 
    } catch (err: any) {
      console.error(err);
      setStatus({ type: 'error', msg: err.message || '训练失败' });
    } finally {
      if (trainInputRef.current) trainInputRef.current.value = '';
    }
  };

  const handleClearData = async () => {
    if (!window.confirm(`⚠️ 警告：确定要清空【${MODEL_CONFIGS[currentModel].name}】的所有训练数据和模型吗？`)) {
      return;
    }
    try {
      setStatus({ type: 'loading', msg: '正在清空数据...' });
      await clearModelData(currentModel);
      setMetrics(null);
      setStatus({ type: 'success', msg: '模型及训练数据已清空。' });
    } catch (err: any) {
      setStatus({ type: 'error', msg: '清空失败: ' + err.message });
    }
  };

  const handleDownloadTrainData = async () => {
     try {
       setStatus({ type: 'loading', msg: '正在准备下载...' });
       await downloadTrainingData(currentModel);
       setStatus({ type: 'success', msg: '累积原始训练数据下载成功。' });
     } catch (err: any) {
       setStatus({ type: 'error', msg: err.message });
     }
  };
  
  const handleDownloadSummary = async () => {
    try {
      await downloadSummary(currentModel);
    } catch (err: any) {
      alert(err.message);
    }
  };

  const onPredictFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    setPredictFileName(file.name);

    try {
      setStatus({ type: 'loading', msg: '初始化预测...' });
      const results = await handlePredict(file, currentModel, (msg) => setStatus({ type: 'loading', msg }));
      setPreviewData(results);
      setStatus({ type: 'success', msg: '预测完成！已基于日均派生特征进行计算。' });
    } catch (err: any) {
      console.error(err);
      setStatus({ type: 'error', msg: err.message || '预测失败' });
      setPreviewData(null);
    } finally {
      if (predictInputRef.current) predictInputRef.current.value = '';
    }
  };

  const handleExport = () => {
    if (!previewData || !predictFileName) return;
    try {
      exportPredictionResults(previewData, predictFileName, currentModel);
      setStatus({ type: 'success', msg: '文件导出成功！' });
    } catch (err: any) {
      setStatus({ type: 'error', msg: '导出失败: ' + err.message });
    }
  };

  const handlePublishToGitHub = async () => {
    if (!ghConfig.token || !ghConfig.owner || !ghConfig.repo) {
      setShowSettings(true);
      setStatus({ type: 'error', msg: '请先配置 GitHub 设置' });
      return;
    }

    if (!window.confirm(`确定要将当前【${MODEL_CONFIGS[currentModel].name}】发布到 GitHub 吗？`)) {
      return;
    }

    try {
      setStatus({ type: 'loading', msg: '正在准备上传数据...' });
      const data = await getModelExportData(currentModel);
      if (!data) throw new Error("无法读取本地模型数据");

      setStatus({ type: 'loading', msg: '正在连接 GitHub...' });
      await uploadToGitHub(ghConfig, data, `Update ${MODEL_CONFIGS[currentModel].name} model and data with feature engineering`);

      setStatus({ type: 'success', msg: `发布成功！GitHub 仓库 ${ghConfig.path} 已更新。` });
    } catch (err: any) {
      setStatus({ type: 'error', msg: '发布失败: ' + err.message });
    }
  };

  const handleCloudPretrain = async () => {
    if (!ghConfig.owner || !ghConfig.repo) {
      setShowSettings(true);
      setStatus({ type: 'error', msg: '请先配置 GitHub 仓库信息' });
      return;
    }

    if (!window.confirm(`确定要从 GitHub 加载【${MODEL_CONFIGS[currentModel].name}】的云端数据吗？`)) {
      return;
    }

    try {
      setStatus({ type: 'loading', msg: '正在从 GitHub 获取数据...' });
      const remoteJson = await fetchFromGitHub(ghConfig);
      
      if (!remoteJson) {
         throw new Error("远程文件为空或不存在");
      }

      setStatus({ type: 'loading', msg: '正在恢复模型与数据...' });
      const newMetrics = await restoreModelFromRemote(currentModel, remoteJson, (msg) => setStatus({ type: 'loading', msg }));
      
      if (newMetrics) {
        setMetrics(newMetrics);
        setStatus({ type: 'success', msg: '云端同步完成！' });
      } else {
        setStatus({ type: 'error', msg: '远程文件中未找到该模型的数据。' });
      }
    } catch (err: any) {
      console.error(err);
      setStatus({ type: 'error', msg: '预训练失败: ' + err.message });
    }
  };

  const saveSettings = (e: React.FormEvent) => {
    e.preventDefault();
    saveGitHubConfig(ghConfig);
    setShowSettings(false);
    setStatus({ type: 'success', msg: 'GitHub 设置已保存' });
  };

  const clearPreview = () => {
    setPreviewData(null);
    setPredictFileName("");
    setStatus({ type: 'idle', msg: '' });
  };

  return (
    <div className="min-h-screen bg-slate-50 pb-12">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-brand-600 p-2 rounded-lg">
              <Activity className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900 tracking-tight">采集量预测系统</h1>
              <p className="text-xs text-gray-500">Feature Engineered RF Predictor</p>
            </div>
          </div>
          <button 
            onClick={() => setShowSettings(!showSettings)}
            className="text-gray-500 hover:text-gray-700 p-2 rounded-full hover:bg-gray-100 transition-colors"
          >
            <Settings size={20} />
          </button>
        </div>
      </header>

      {/* GitHub Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full p-6 space-y-4">
            <div className="flex justify-between items-center border-b pb-3">
              <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                <Github size={20} /> GitHub 配置
              </h3>
              <button onClick={() => setShowSettings(false)} className="text-gray-400 hover:text-gray-600">
                <X size={20} />
              </button>
            </div>
            <form onSubmit={saveSettings} className="space-y-4">
               <div>
                <label className="block text-sm font-medium text-gray-700">Repo Owner</label>
                <input 
                  type="text" 
                  required
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-brand-500 focus:ring-brand-500 sm:text-sm p-2 border"
                  value={ghConfig.owner}
                  onChange={e => setGhConfig({...ghConfig, owner: e.target.value})}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Repository Name</label>
                <input 
                  type="text" 
                  required
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-brand-500 focus:ring-brand-500 sm:text-sm p-2 border"
                  value={ghConfig.repo}
                  onChange={e => setGhConfig({...ghConfig, repo: e.target.value})}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">File Path</label>
                <input 
                  type="text" 
                  required
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-brand-500 focus:ring-brand-500 sm:text-sm p-2 border"
                  value={ghConfig.path}
                  onChange={e => setGhConfig({...ghConfig, path: e.target.value})}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Personal Access Token (PAT)</label>
                <input 
                  type="password" 
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-brand-500 focus:ring-brand-500 sm:text-sm p-2 border"
                  value={ghConfig.token}
                  onChange={e => setGhConfig({...ghConfig, token: e.target.value})}
                />
              </div>
              <div className="flex justify-end gap-3 pt-2">
                <Button type="button" variant="outline" size="sm" onClick={() => setShowSettings(false)}>取消</Button>
                <Button type="submit" variant="primary" size="sm">保存配置</Button>
              </div>
            </form>
          </div>
        </div>
      )}

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        
        {/* Model Selector Tabs */}
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            {Object.values(ModelType).map((type) => (
              <button
                key={type}
                onClick={() => setCurrentModel(type)}
                className={`
                  whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm flex items-center gap-2 transition-colors
                  ${currentModel === type
                    ? 'border-brand-500 text-brand-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}
                `}
              >
                {MODEL_CONFIGS[type].name}
              </button>
            ))}
          </nav>
        </div>

        {/* Status Banner */}
        {status.type !== 'idle' && (
          <div className={`rounded-md p-4 border ${
            status.type === 'error' ? 'bg-red-50 text-red-700 border-red-200' : 
            status.type === 'success' ? 'bg-green-50 text-green-700 border-green-200' : 
            'bg-blue-50 text-blue-700 border-blue-200'
          }`}>
            <div className="flex items-center">
              {status.type === 'error' && <AlertCircle className="h-5 w-5 mr-3 text-red-400" />}
              {status.type === 'success' && <CheckCircle className="h-5 w-5 mr-3 text-green-400" />}
              {status.type === 'loading' && <Activity className="h-5 w-5 mr-3 text-blue-400 animate-pulse" />}
              <p className="text-sm font-medium">{status.msg}</p>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          <div className="lg:col-span-1 space-y-6">
            <Card title="模型状态">
              {isInitializing ? (
                <div className="py-12 flex justify-center">
                  <Activity className="h-8 w-8 text-brand-300 animate-spin" />
                </div>
              ) : metrics ? (
                <div className="space-y-6">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm font-medium text-gray-500">R²</p>
                      <p className="mt-1 text-2xl font-semibold text-gray-900">{metrics.r2.toFixed(4)}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-500">MAE</p>
                      <p className="mt-1 text-2xl font-semibold text-gray-900">{metrics.mae.toFixed(2)}</p>
                    </div>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">累积样本量</p>
                    <p className="mt-1 text-3xl font-semibold text-gray-900">{metrics.sampleSize.toLocaleString()}</p>
                  </div>
                  <div className="pt-4 border-t border-gray-100">
                    <p className="text-xs text-gray-400">上次更新: {metrics.lastUpdated}</p>
                  </div>
                  <Button 
                    variant="secondary" 
                    className="w-full" 
                    onClick={handleDownloadSummary}
                    icon={<Download size={16} />}
                  >
                    下载训练摘要
                  </Button>
                </div>
              ) : (
                <div className="text-center py-10">
                  <div className="bg-gray-100 rounded-full h-12 w-12 flex items-center justify-center mx-auto mb-3">
                    <BarChart2 className="text-gray-400" />
                  </div>
                  <p className="text-gray-500">该模型尚未训练</p>
                </div>
              )}
            </Card>

            <div className="bg-blue-50 rounded-lg p-4 border border-blue-100 text-sm">
              <h4 className="font-bold mb-2 flex items-center gap-2 text-blue-800">
                <AlertCircle size={16}/> 特征派生说明
              </h4>
              <p className="text-blue-700 text-xs leading-relaxed">
                系统会自动将“笔记数/点赞数”等总量转换为“日均值”进行建模。原始总量字段仅用于展示，不直接参与回归计算。
              </p>
            </div>
          </div>

          <div className="lg:col-span-2 space-y-6">
            
            <Card title="模型训练 (累积学习)">
              <div className="space-y-4">
                <p className="text-sm text-gray-600">
                  上传 Excel 自动合并历史数据。系统会执行特征工程并训练 200 棵随机森林回归树。
                </p>
                <div className="flex flex-wrap items-center gap-3">
                  <input
                    type="file"
                    ref={trainInputRef}
                    onChange={onTrainFileChange}
                    accept=".xlsx,.xls,.csv"
                    className="hidden"
                  />
                  <Button 
                    onClick={() => trainInputRef.current?.click()} 
                    isLoading={status.type === 'loading'}
                    disabled={isInitializing}
                    icon={<Upload size={18} />}
                  >
                    上传训练数据
                  </Button>
                  <Button 
                     onClick={handleCloudPretrain}
                     disabled={isInitializing}
                     variant="secondary"
                     icon={<CloudDownload size={18} />}
                  >
                    云端同步
                  </Button>
                  <Button 
                    variant="outline" 
                    size="md" 
                    onClick={generateTrainTemplate} 
                    icon={<FileSpreadsheet size={16}/>}
                  >
                    模板
                  </Button>
                </div>
                
                <div className="pt-4 border-t border-gray-100 mt-4 flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-3">
                     <Button
                      variant="secondary"
                      size="sm"
                      onClick={handleDownloadTrainData}
                      disabled={!metrics || isInitializing}
                      icon={<Database size={14}/>}
                    >
                      原始数据导出
                    </Button>
                    <Button
                      variant="primary"
                      size="sm"
                      onClick={handlePublishToGitHub}
                      disabled={!metrics || isInitializing}
                      icon={<CloudUpload size={14}/>}
                      className="bg-gray-800 hover:bg-gray-900"
                    >
                      发布云端
                    </Button>
                  </div>
                  <Button
                    variant="danger"
                    size="sm"
                    onClick={handleClearData}
                    disabled={!metrics || isInitializing}
                    icon={<Trash2 size={14}/>}
                    className="bg-red-50 text-red-600 hover:bg-red-100 border border-red-200"
                  >
                    重置模型
                  </Button>
                </div>
              </div>
            </Card>

            <Card title="采集量预测">
               <div className="space-y-4">
                <p className="text-sm text-gray-600">
                   上传预测文件。系统将自动提取日均特征进行 80% 置信区间预测。
                </p>
                <div className="flex flex-wrap items-center gap-3">
                   <input
                    type="file"
                    ref={predictInputRef}
                    onChange={onPredictFileChange}
                    accept=".xlsx,.xls,.csv"
                    className="hidden"
                  />
                  <Button 
                    variant="primary"
                    className="bg-purple-600 hover:bg-purple-700 focus:ring-purple-500"
                    onClick={() => predictInputRef.current?.click()}
                    isLoading={status.type === 'loading'}
                    disabled={!metrics || isInitializing}
                    icon={<Activity size={18} />}
                  >
                    开始预测
                  </Button>
                   <Button 
                    variant="outline" 
                    size="md" 
                    onClick={generatePredictionTemplate} 
                    icon={<FileSpreadsheet size={16}/>}
                  >
                    预测模板
                  </Button>
                </div>
              </div>
            </Card>

            {previewData && (
              <Card title="预测结果预览" className="border-brand-200 ring-4 ring-brand-50">
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <p className="text-sm text-gray-500">
                      共 <span className="font-bold text-gray-900">{previewData.length}</span> 条结果
                    </p>
                    <div className="flex gap-2">
                       <Button 
                        variant="outline" 
                        size="sm" 
                        onClick={clearPreview}
                        icon={<X size={16} />}
                      >
                        清除
                      </Button>
                      <Button 
                        variant="primary" 
                        size="sm" 
                        onClick={handleExport}
                        icon={<Save size={16} />}
                      >
                        导出 Excel
                      </Button>
                    </div>
                  </div>

                  <div className="overflow-x-auto border rounded-lg max-h-96">
                    <table className="min-w-full divide-y divide-gray-200 text-sm">
                      <thead className="bg-gray-50 sticky top-0">
                        <tr>
                          {INPUT_FEATURES.map(f => (
                            <th key={f} className="px-3 py-2 text-left font-medium text-gray-500 uppercase tracking-wider">{f}</th>
                          ))}
                          <th className="px-3 py-2 text-left font-bold text-brand-600 bg-brand-50 uppercase tracking-wider">预测采集量</th>
                          <th className="px-3 py-2 text-left font-medium text-gray-500 uppercase tracking-wider">预测区间 (80%)</th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {previewData.map((row, idx) => (
                          <tr key={idx} className="hover:bg-gray-50">
                            {INPUT_FEATURES.map(f => (
                              <td key={f} className="px-3 py-2 whitespace-nowrap text-gray-700">{row[f]}</td>
                            ))}
                            <td className="px-3 py-2 whitespace-nowrap font-bold text-brand-600 bg-brand-50/30">{row['预测采集量']}</td>
                            <td className="px-3 py-2 whitespace-nowrap text-gray-500">
                              [{row['预测下限']} - {row['预测上限']}]
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </Card>
            )}

          </div>
        </div>
      </main>
    </div>
  );
};

export default App;
