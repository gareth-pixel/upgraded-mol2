
import React, { useState, useEffect, useRef } from 'react';
import { Download, Upload, BarChart2, FileSpreadsheet, Activity, AlertCircle, CheckCircle, X, Save, Trash2, Database, Github, Settings, CloudUpload, CloudDownload, ChevronDown, ChevronUp, ShieldCheck } from 'lucide-react';
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

  // Guardrail state
  const [showAdvance, setShowAdvance] = useState(false);
  const [guardrailEnabled, setGuardrailEnabled] = useState(true);
  const [lowPercent, setLowPercent] = useState(30);
  const [highPercent, setHighPercent] = useState(170);

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
      const modeDesc = currentModel === ModelType.RECALL ? "已启用日均强度学习与天数放大" : "已启用总量直接推理";
      setStatus({ type: 'success', msg: `模型训练完成！${modeDesc}。` });
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
      const results = await handlePredict(
        file, 
        currentModel, 
        (msg) => setStatus({ type: 'loading', msg }),
        {
          enabled: guardrailEnabled,
          lowPercent,
          highPercent
        }
      );
      setPreviewData(results);
      setStatus({ type: 'success', msg: `预测完成！` });
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

  const isDailyMode = currentModel === ModelType.RECALL;

  return (
    <div className="min-h-screen bg-slate-50 pb-12">
      <header className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-brand-600 p-2 rounded-lg">
              <Activity className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900 tracking-tight">采集量预测系统 v2.4</h1>
              <p className="text-xs text-gray-500">Hybrid Intensity & Total Yield Predictor</p>
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
               <input placeholder="Repo Owner" className="w-full p-2 border rounded text-sm" value={ghConfig.owner} onChange={e => setGhConfig({...ghConfig, owner: e.target.value})} />
               <input placeholder="Repo Name" className="w-full p-2 border rounded text-sm" value={ghConfig.repo} onChange={e => setGhConfig({...ghConfig, repo: e.target.value})} />
               <input placeholder="File Path" className="w-full p-2 border rounded text-sm" value={ghConfig.path} onChange={e => setGhConfig({...ghConfig, path: e.target.value})} />
               <input placeholder="Token" type="password" className="w-full p-2 border rounded text-sm" value={ghConfig.token} onChange={e => setGhConfig({...ghConfig, token: e.target.value})} />
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
              <div className="space-y-2 text-slate-700">
                <p><b>XGBoost 推理引擎：</b></p>
                {isDailyMode ? (
                  <p className="bg-white/50 p-2 rounded border border-brand-200">
                    <span className="text-brand-700 font-bold">回溯模式（强度学习）：</span><br/>
                    系统自动将数据对齐至“日均”量级，预测时基于学习到的日均系数乘以【采集天数】进行线性放大。
                    <span className="text-blue-700 font-semibold italic block mt-1">支持：通过中位数比值法建立基线限幅，防止离群值干扰。</span>
                  </p>
                ) : (
                  <p className="bg-white/50 p-2 rounded border border-brand-200">
                    <span className="text-brand-700 font-bold">在线模式（总量推理）：</span><br/>
                    直接建立指标总量与产出总量的映射。不涉及天数除法，更贴合单天的业务场景。
                    <span className="text-slate-500 italic block mt-1">注：此模式下不启用基线限幅。</span>
                  </p>
                )}
              </div>
            </div>
          </div>

          <div className="lg:col-span-2 space-y-6">
            <Card title="数据学习">
              <div className="space-y-4">
                <p className="text-sm text-gray-600">上传 Excel 训练数据。系统将根据{isDailyMode ? '日均强度' : '总量级'}提取业务产出系数并拟合 XGBoost。</p>
                <div className="flex flex-wrap gap-3">
                  <input type="file" ref={trainInputRef} onChange={onTrainFileChange} accept=".xlsx,.xls" className="hidden" />
                  <Button onClick={() => trainInputRef.current?.click()} icon={<Upload size={18} />}>上传并训练</Button>
                  <Button onClick={handleCloudPretrain} variant="secondary" icon={<CloudDownload size={18} />}>云端同步</Button>
                  <Button variant="outline" onClick={generateTrainTemplate} icon={<FileSpreadsheet size={16}/>}>训练模板</Button>
                </div>
                <div className="pt-4 border-t flex justify-between gap-3">
                  <div className="flex gap-2">
                    <Button variant="secondary" size="sm" onClick={handleDownloadTrainData} icon={<Database size={14}/>}>导出数据库</Button>
                    <Button variant="primary" size="sm" onClick={handlePublishToGitHub} icon={<CloudUpload size={14}/>} className="bg-gray-800">发布至云端</Button>
                  </div>
                  <Button variant="danger" size="sm" onClick={handleClearData} icon={<Trash2 size={14}/>} className="bg-red-600 text-white border-red-200">重置模型</Button>
                </div>
              </div>
            </Card>

            <Card title="流量预测">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <p className="text-sm text-gray-600">基于已学习的产出路径进行推理。{isDailyMode ? '采集天数将作为线性放大系数。' : '直接基于总量进行映射。'}</p>
                  {isDailyMode && (
                    <button 
                      onClick={() => setShowAdvance(!showAdvance)}
                      className="text-xs flex items-center gap-1 text-brand-600 hover:text-brand-700 font-medium"
                    >
                      {showAdvance ? <ChevronUp size={14}/> : <ChevronDown size={14}/>}
                      策略配置 {guardrailEnabled && <ShieldCheck size={12} className="inline"/>}
                    </button>
                  )}
                </div>

                {isDailyMode && showAdvance && (
                  <div className="p-4 bg-slate-50 border rounded-lg space-y-4 animate-in fade-in slide-in-from-top-2 duration-200">
                    <div className="flex items-center gap-2">
                      <input 
                        type="checkbox" 
                        id="guardrail_toggle" 
                        checked={guardrailEnabled}
                        onChange={(e) => setGuardrailEnabled(e.target.checked)}
                        className="w-4 h-4 text-brand-600 rounded focus:ring-brand-500"
                      />
                      <label htmlFor="guardrail_toggle" className="text-sm font-semibold text-gray-700 flex items-center gap-1">
                        启用稳健基线限幅 <span className="text-xs font-normal text-gray-400">(强制预测值回归业务基线)</span>
                      </label>
                    </div>
                    
                    {guardrailEnabled && (
                      <div className="grid grid-cols-2 gap-4 pl-6">
                        <div className="space-y-1">
                          <label className="text-xs text-gray-500">限幅下限 (%)</label>
                          <div className="relative">
                            <input 
                              type="number" 
                              value={lowPercent} 
                              onChange={(e) => setLowPercent(Number(e.target.value))}
                              className="w-full p-2 border rounded text-sm pr-8"
                            />
                            <span className="absolute right-2 top-2 text-gray-400 text-xs">%</span>
                          </div>
                        </div>
                        <div className="space-y-1">
                          <label className="text-xs text-gray-500">限幅上限 (%)</label>
                          <div className="relative">
                            <input 
                              type="number" 
                              value={highPercent} 
                              onChange={(e) => setHighPercent(Number(e.target.value))}
                              className="w-full p-2 border rounded text-sm pr-8"
                            />
                            <span className="absolute right-2 top-2 text-gray-400 text-xs">%</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                <div className="flex gap-3">
                  <input type="file" ref={predictInputRef} onChange={onPredictFileChange} accept=".xlsx,.xls" className="hidden" />
                  <Button variant="primary" className="bg-brand-600 hover:bg-brand-700" onClick={() => predictInputRef.current?.click()} icon={<Activity size={18} />}>执行推理预测</Button>
                  <Button variant="outline" onClick={generatePredictionTemplate} icon={<FileSpreadsheet size={16}/>}>下载模板</Button>
                </div>
              </div>
            </Card>

            {previewData && (
              <Card title="预测结果预览" className="border-brand-200 ring-4 ring-brand-50">
                <div className="space-y-4">
                  <div className="flex justify-between items-center text-sm">
                    <div className="flex gap-4">
                       <p>样本总数: <span className="font-bold">{previewData.length}</span></p>
                       {previewData.some(r => r._IS_CLIPPED) && (
                         <p className="text-amber-600 flex items-center gap-1 text-xs">
                           <ShieldCheck size={14}/> 发现受限幅修正的数据
                         </p>
                       )}
                    </div>
                    <div className="flex gap-2">
                      <Button variant="outline" size="sm" onClick={() => setPreviewData(null)} icon={<X size={16} />}>清除</Button>
                      <Button variant="primary" size="sm" onClick={handleExport} icon={<Save size={16} />}>保存 Excel</Button>
                    </div>
                  </div>
                  <div className="overflow-x-auto border rounded-lg max-h-96">
                    <table className="min-w-full divide-y divide-gray-200 text-xs">
                      <thead className="bg-gray-50 sticky top-0">
                        <tr>
                          {INPUT_FEATURES.map(f => <th key={f} className="px-3 py-2 text-left font-medium text-gray-500 uppercase">{f}</th>)}
                          <th className="px-3 py-2 text-left font-bold text-brand-600 bg-brand-50">预测采集量</th>
                          <th className="px-3 py-2 text-left font-medium text-gray-500 text-center">80% 置信区间</th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {previewData.map((row, idx) => (
                          <tr key={idx} className="hover:bg-gray-50">
                            {INPUT_FEATURES.map(f => <td key={f} className="px-3 py-2">{row[f]}</td>)}
                            <td className={`px-3 py-2 font-bold ${row._IS_CLIPPED ? 'text-amber-600 bg-amber-50' : 'text-brand-600 bg-brand-50/30'}`}>
                              <div className="flex items-center gap-1">
                                {row['预测采集量']}
                                {row._IS_CLIPPED && (
                                  <span title="已触发基线限幅修正">
                                    <ShieldCheck size={12} className="text-amber-500" />
                                  </span>
                                )}
                              </div>
                            </td>
                            <td className="px-3 py-2 text-gray-500 text-center">
                              <span className="bg-gray-100 px-2 py-0.5 rounded-full">
                                {row['预测下限']} ~ {row['预测上限']}
                              </span>
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
