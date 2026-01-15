
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
  const [previewData, setPreviewData] = useState<DataRow[] | null>(null);
  const [predictFileName, setPredictFileName] = useState<string>("");
  const [showSettings, setShowSettings] = useState(false);
  const [ghConfig, setGhConfig] = useState<GitHubConfig>({ token: '', owner: '', repo: '', path: 'public/data/model_result.json' });

  const trainInputRef = useRef<HTMLInputElement>(null);
  const predictInputRef = useRef<HTMLInputElement>(null);

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
        console.error(e);
      } finally {
        setIsInitializing(false);
      }
    };
    loadMetrics();
  }, [currentModel]);

  useEffect(() => {
    const saved = getGitHubConfig();
    if (saved) setGhConfig(saved);
  }, []);

  const onTrainFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.length) return;
    try {
      setStatus({ type: 'loading', msg: '初始化训练...' });
      const newMetrics = await handleTrain(e.target.files[0], currentModel, (msg) => setStatus({ type: 'loading', msg }));
      setMetrics(newMetrics);
      setStatus({ type: 'success', msg: `模型训练完成！${currentModel === ModelType.ONLINE ? '已应用天数线性放大逻辑。' : ''}` });
      setPreviewData(null); 
    } catch (err: any) {
      setStatus({ type: 'error', msg: err.message || '训练失败' });
    } finally {
      if (trainInputRef.current) trainInputRef.current.value = '';
    }
  };

  const handleClearData = async () => {
    if (!window.confirm(`确定要清空【${MODEL_CONFIGS[currentModel].name}】吗？`)) return;
    try {
      setStatus({ type: 'loading', msg: '清空数据...' });
      await clearModelData(currentModel);
      setMetrics(null);
      setStatus({ type: 'success', msg: '模型已重置。' });
    } catch (err: any) {
      setStatus({ type: 'error', msg: err.message });
    }
  };

  const handleDownloadTrainData = async () => {
     try {
       await downloadTrainingData(currentModel);
       setStatus({ type: 'success', msg: '训练数据下载成功。' });
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
    if (!e.target.files?.length) return;
    const file = e.target.files[0];
    setPredictFileName(file.name);
    try {
      setStatus({ type: 'loading', msg: '正在预测...' });
      const results = await handlePredict(file, currentModel, (msg) => setStatus({ type: 'loading', msg }));
      setPreviewData(results);
      setStatus({ type: 'success', msg: '预测完成！' });
    } catch (err: any) {
      setStatus({ type: 'error', msg: err.message || '预测失败' });
    } finally {
      if (predictInputRef.current) predictInputRef.current.value = '';
    }
  };

  const handleExport = () => {
    if (!previewData) return;
    try {
      exportPredictionResults(previewData, predictFileName, currentModel);
      setStatus({ type: 'success', msg: '导出成功！' });
    } catch (err: any) {
      setStatus({ type: 'error', msg: err.message });
    }
  };

  const handlePublishToGitHub = async () => {
    if (!ghConfig.token || !ghConfig.owner || !ghConfig.repo) {
      setShowSettings(true);
      return;
    }
    try {
      setStatus({ type: 'loading', msg: '发布云端...' });
      const data = await getModelExportData(currentModel);
      if (!data) throw new Error("无数据");
      await uploadToGitHub(ghConfig, data, `Update ${MODEL_CONFIGS[currentModel].name} model parameters`);
      setStatus({ type: 'success', msg: `发布成功！` });
    } catch (err: any) {
      setStatus({ type: 'error', msg: err.message });
    }
  };

  const handleCloudPretrain = async () => {
    try {
      setStatus({ type: 'loading', msg: '同步云端...' });
      const remoteJson = await fetchFromGitHub(ghConfig);
      if (!remoteJson) throw new Error("远程文件未找到或为空");
      const newMetrics = await restoreModelFromRemote(currentModel, remoteJson, (msg) => setStatus({ type: 'loading', msg }));
      if (newMetrics) {
        setMetrics(newMetrics);
        setStatus({ type: 'success', msg: '同步完成！' });
      } else {
        setStatus({ type: 'error', msg: '云端未发现当前子模型的数据' });
      }
    } catch (err: any) {
      setStatus({ type: 'error', msg: err.message });
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 pb-12">
      <header className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-brand-600 p-2 rounded-lg">
              <Activity className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900 tracking-tight">采集量预测系统 v2.0</h1>
              <p className="text-xs text-gray-500">Multi-Model Volume Forecaster</p>
            </div>
          </div>
          <button onClick={() => setShowSettings(!showSettings)} className="text-gray-500 hover:text-gray-700 p-2 rounded-full hover:bg-gray-100"><Settings size={20} /></button>
        </div>
      </header>

      {showSettings && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full p-6 space-y-4">
            <h3 className="text-lg font-semibold flex items-center gap-2 border-b pb-2"><Github size={20} /> GitHub 同步配置</h3>
            <div className="space-y-3">
               <input placeholder="Repo Owner" className="w-full p-2 border rounded" value={ghConfig.owner} onChange={e => setGhConfig({...ghConfig, owner: e.target.value})} />
               <input placeholder="Repo Name" className="w-full p-2 border rounded" value={ghConfig.repo} onChange={e => setGhConfig({...ghConfig, repo: e.target.value})} />
               <input placeholder="File Path" className="w-full p-2 border rounded" value={ghConfig.path} onChange={e => setGhConfig({...ghConfig, path: e.target.value})} />
               <input placeholder="Token" type="password" className="w-full p-2 border rounded" value={ghConfig.token} onChange={e => setGhConfig({...ghConfig, token: e.target.value})} />
            </div>
            <div className="flex justify-end gap-2">
              <Button variant="outline" size="sm" onClick={() => setShowSettings(false)}>取消</Button>
              <Button variant="primary" size="sm" onClick={() => { saveGitHubConfig(ghConfig); setShowSettings(false); }}>保存配置</Button>
            </div>
          </div>
        </div>
      )}

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            {Object.values(ModelType).map((type) => (
              <button key={type} onClick={() => setCurrentModel(type)} className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${currentModel === type ? 'border-brand-500 text-brand-600' : 'border-transparent text-gray-500 hover:text-gray-700'}`}>
                {MODEL_CONFIGS[type].name}
              </button>
            ))}
          </nav>
        </div>

        {status.type !== 'idle' && (
          <div className={`rounded-md p-4 border ${status.type === 'error' ? 'bg-red-50 text-red-700 border-red-200' : status.type === 'success' ? 'bg-green-50 text-green-700 border-green-200' : 'bg-blue-50 text-blue-700 border-blue-200'}`}>
            <div className="flex items-center">
              {status.type === 'error' && <AlertCircle className="h-5 w-5 mr-3" />}
              {status.type === 'success' && <CheckCircle className="h-5 w-5 mr-3" />}
              {status.type === 'loading' && <Activity className="h-5 w-5 mr-3 animate-pulse" />}
              <p className="text-sm font-medium">{status.msg}</p>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-1 space-y-6">
            <Card title="模型状态">
              {isInitializing ? <div className="py-12 flex justify-center"><Activity className="animate-spin text-brand-300" /></div> : metrics ? (
                <div className="space-y-6">
                  <div className="grid grid-cols-2 gap-4">
                    <div><p className="text-sm text-gray-500">R² (回归拟合度)</p><p className="text-2xl font-bold">{metrics.r2.toFixed(4)}</p></div>
                    <div><p className="text-sm text-gray-500">MAE (平均误差)</p><p className="text-2xl font-bold">{metrics.mae.toFixed(2)}</p></div>
                  </div>
                  <div><p className="text-sm text-gray-500">训练样本量</p><p className="text-3xl font-bold">{metrics.sampleSize.toLocaleString()}</p></div>
                  <Button variant="secondary" className="w-full" onClick={handleDownloadSummary} icon={<Download size={16} />}>下载模型详情</Button>
                </div>
              ) : <div className="text-center py-10 text-gray-400"><BarChart2 className="mx-auto mb-2" />模型待训练</div>}
            </Card>
            <div className="bg-brand-50 p-4 rounded-lg border border-brand-100 text-xs text-brand-800">
              <h4 className="font-bold flex items-center gap-2 mb-1"><AlertCircle size={14}/> 算法逻辑说明</h4>
              {currentModel === ModelType.ONLINE ? (
                <div className="space-y-2">
                  <p>当前使用 <b>梯度提升回归树 (XGBoost)</b>。</p>
                  <p className="text-blue-700 font-semibold underline decoration-blue-300">专项优化：采集天数不参与内部权重，通过学习“日均采集量”并在最后按天数线性乘回，确保长期预测的线性逻辑。</p>
                </div>
              ) : (
                <p>当前使用 <b>岭回归 (Ridge Regression)</b>。采集天数作为普通线性特征参与全局权重计算，适合处理小样本稳定性需求。</p>
              )}
            </div>
          </div>

          <div className="lg:col-span-2 space-y-6">
            <Card title="数据学习">
              <div className="space-y-4">
                <p className="text-sm text-gray-600">上传 Excel 训练数据。系统将自动完成日均特征派生与模型拟合。</p>
                <div className="flex flex-wrap gap-3">
                  <input type="file" ref={trainInputRef} onChange={onTrainFileChange} accept=".xlsx,.xls" className="hidden" />
                  <Button onClick={() => trainInputRef.current?.click()} icon={<Upload size={18} />}>上传并训练</Button>
                  <Button onClick={handleCloudPretrain} variant="secondary" icon={<CloudDownload size={18} />}>从云端同步</Button>
                  <Button variant="outline" onClick={generateTrainTemplate} icon={<FileSpreadsheet size={16}/>}>训练模板</Button>
                </div>
                <div className="pt-4 border-t flex justify-between gap-3">
                  <div className="flex gap-2">
                    <Button variant="secondary" size="sm" onClick={handleDownloadTrainData} icon={<Database size={14}/>}>导出数据库</Button>
                    <Button variant="primary" size="sm" onClick={handlePublishToGitHub} icon={<CloudUpload size={14}/>} className="bg-gray-800">发布至云端</Button>
                  </div>
                  <Button variant="danger" size="sm" onClick={handleClearData} icon={<Trash2 size={14}/>} className="bg-red-50 text-red-600 border-red-200">重置模型</Button>
                </div>
              </div>
            </Card>

            <Card title="流量预测">
              <div className="space-y-4">
                <p className="text-sm text-gray-600">基于已学习的权重/树路径进行推理，输出 80% 置信区间预测结果。</p>
                <div className="flex gap-3">
                  <input type="file" ref={predictInputRef} onChange={onPredictFileChange} accept=".xlsx,.xls" className="hidden" />
                  <Button variant="primary" className="bg-purple-600 hover:bg-purple-700" onClick={() => predictInputRef.current?.click()} icon={<Activity size={18} />}>开始预测</Button>
                  <Button variant="outline" onClick={generatePredictionTemplate} icon={<FileSpreadsheet size={16}/>}>预测模板</Button>
                </div>
              </div>
            </Card>

            {previewData && (
              <Card title="预测预览" className="border-brand-200 ring-4 ring-brand-50">
                <div className="space-y-4">
                  <div className="flex justify-between items-center text-sm">
                    <p>预测样本: <span className="font-bold">{previewData.length}</span></p>
                    <div className="flex gap-2">
                      <Button variant="outline" size="sm" onClick={() => setPreviewData(null)} icon={<X size={16} />}>清除</Button>
                      <Button variant="primary" size="sm" onClick={handleExport} icon={<Save size={16} />}>导出结果</Button>
                    </div>
                  </div>
                  <div className="overflow-x-auto border rounded-lg max-h-96">
                    <table className="min-w-full divide-y divide-gray-200 text-xs">
                      <thead className="bg-gray-50 sticky top-0">
                        <tr>
                          {INPUT_FEATURES.map(f => <th key={f} className="px-3 py-2 text-left font-medium text-gray-500 uppercase">{f}</th>)}
                          <th className="px-3 py-2 text-left font-bold text-brand-600 bg-brand-50">预测采集量</th>
                          <th className="px-3 py-2 text-left font-medium text-gray-500">置信区间 (80%)</th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {previewData.map((row, idx) => (
                          <tr key={idx} className="hover:bg-gray-50">
                            {INPUT_FEATURES.map(f => <td key={f} className="px-3 py-2">{row[f]}</td>)}
                            <td className="px-3 py-2 font-bold text-brand-600 bg-brand-50/30">{row['预测采集量']}</td>
                            <td className="px-3 py-2 text-gray-500">[{row['预测下限']} - {row['预测上限']}]</td>
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
