import {useEffect, useRef, useState, type FormEvent, type ReactNode} from 'react';
import {createRoot} from 'react-dom/client';
import {io, type Socket} from 'socket.io-client';
import {Activity, ArrowDownToLine, ArrowRight, ArrowUpRight, Check, ChevronDown, CircleHelp, Code2, Cpu, FlaskConical, History, Layers3, LoaderCircle, Moon, Play, Plus, Radio, RotateCcw, Settings2, Square, Sun, Terminal, Waypoints, WifiOff, X, Zap} from 'lucide-react';
import {applyEvent, type Catalog, type Config, type Generation, type HFInspect, type Metric, type Run, type RunEvent, type Summary} from './types';
import './styles.css';

const HF_TASK = 'hf_generic.HFGenericTask';
const initial: Config = {task: 'game_2048.RL2048', model_id: 'gemma', action: 'train', mode: 'training', steps: 200, group: 4, lr: .000005, max_tokens: 512, temp: .9, lora_rank: 8, lora_layers: 8, allow_unsafe_code: false, task_params: {}};
const isActive = (status?: string) => status === 'running' || status === 'stopping';
const number = (value?: number | null, digits = 3) => value == null || !Number.isFinite(value) ? '—' : value.toFixed(digits);
const label = (value: string) => value.split('_').map(w => w[0].toUpperCase() + w.slice(1)).join(' ');
const taskName = (task?: string) => task?.startsWith('game') ? '2048 strategy' : task?.includes('hf_generic') ? 'Custom HF dataset' : 'MBPP unit tests';
const modelName = (value?: string) => value === 'gpt-oss' ? 'GPT-OSS' : value === 'gemma' ? 'Gemma' : value === 'qwen' ? 'Qwen' : value?.split('/').at(-1) || 'Model';
function elapsed(run: Summary | null, now: number) {if (!run) return '00:00'; const seconds = Math.max(0, Math.floor(((run.finished_at ? Date.parse(run.finished_at) : now) - Date.parse(run.created_at)) / 1000)); return `${Math.floor(seconds/60).toString().padStart(2,'0')}:${(seconds%60).toString().padStart(2,'0')}`;}

function App() {
  const [connected, setConnected] = useState(false);
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [runs, setRuns] = useState<Summary[]>([]);
  const [run, setRun] = useState<Run | null>(null);
  const [config, setConfig] = useState<Config>(initial);
  const [page, setPage] = useState('workspace');
  const [tab, setTab] = useState('completions');
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const [clock, setClock] = useState(Date.now());
  const [follow, setFollow] = useState(true);
  const [generationId, setGenerationId] = useState<string | null>(null);
  const [chart, setChart] = useState<'mean_reward' | 'loss'>('mean_reward');
  const [help, setHelp] = useState(false);
  const [hfInspect, setHfInspect] = useState<HFInspect | null>(null);
  const [hfInspecting, setHfInspecting] = useState(false);
  const [hfDatasetMode, setHfDatasetMode] = useState<'hf' | 'upload'>('hf');
  const [hfUploading, setHfUploading] = useState(false);
  const [theme, setTheme] = useState<'dark'|'light'>(() => (localStorage.getItem('reinforce-theme') as 'dark'|'light') || 'dark');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('reinforce-theme', theme);
  }, [theme]);

  function toggleTheme() {
    document.documentElement.classList.add('theme-transitioning');
    setTimeout(() => document.documentElement.classList.remove('theme-transitioning'), 300);
    setTheme(t => t === 'dark' ? 'light' : 'dark');
  }
  const socket = useRef<Socket | null>(null);
  const selected = useRef<string | null>(null);
  const output = useRef<HTMLPreElement>(null);
  const form = useRef<HTMLFormElement>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetch('/api/catalog', {signal: controller.signal}).then(r => {if (!r.ok) throw Error('Backend unavailable'); return r.json();}).then(setCatalog).catch(e => {if (e.name !== 'AbortError') setError(e.message);});
    const s = io({autoConnect: false, reconnection: true}); socket.current = s;
    s.on('connect', () => {setConnected(true); setError(''); if (selected.current) s.emit('watch_run', {run_id: selected.current});});
    s.on('disconnect', () => setConnected(false));
    s.on('connect_error', () => {setConnected(false); setError('Cannot reach the backend. Reconnecting automatically…');});
    s.on('run_list', (items: Summary[]) => {setRuns(items); if (!selected.current && items.length) {selected.current = items[0].id; s.emit('watch_run', {run_id: items[0].id});}});
    s.on('run_snapshot', (snapshot: Run) => {if (selected.current && selected.current !== snapshot.id) return; selected.current = snapshot.id; setRun(old => old?.id === snapshot.id && old.seq > snapshot.seq ? old : snapshot);});
    s.on('run_event', (event: RunEvent) => setRun(old => old ? applyEvent(old, event) : old));
    s.connect();
    const timer = window.setInterval(() => setClock(Date.now()), 1000);
    return () => {controller.abort(); s.disconnect(); socket.current = null; clearInterval(timer);};
  }, []);
  const latest = run?.generations.at(-1);
  const generation = run?.generations.find(g => g.generation_id === generationId) ?? latest;
  useEffect(() => {if (follow && output.current) output.current.scrollTop = output.current.scrollHeight;}, [generation?.text, run?.logs, follow, tab]);
  const active = runs.find(r => isActive(r.status));
  const metric = run?.metrics.at(-1);
  const total = metric?.steps ?? (run?.config.mode === 'demo' ? Math.min(run.config.steps, 12) : run?.config.steps ?? config.steps);
  const tracksSteps = !run || run.config.mode === 'demo' || run.config.action === 'train';
  const progress = tracksSteps ? Math.min(100, ((metric?.step ?? 0) / total) * 100) : run?.status === 'completed' ? 100 : 0;

  function watch(id: string) {selected.current = id; setGenerationId(null); setPage('workspace'); socket.current?.timeout(10000).emit('watch_run', {run_id: id}, (err: Error | null, data: {ok: boolean; error?: string}) => {if (err || !data?.ok) setError(data?.error || 'Could not open the run');});}
  function launch(demo = false) {
    setError(''); setPending(true); setGenerationId(null); setTab('completions');
    const previous = selected.current; selected.current = null; setRun(null);
    const payload = {...config, mode: demo ? 'demo' : 'training'};
    socket.current?.timeout(12000).emit('start_run', payload, (err: Error | null, data: {ok: boolean; error?: string; run_id?: string}) => {
      setPending(false);
      if (err || !data?.ok) {selected.current = previous; setError(data?.error || 'Start request timed out. Check run history before retrying.'); if (previous) watch(previous);}
      else if (data.run_id) {selected.current = data.run_id; watch(data.run_id); setPage('workspace');}
    });
  }
  function stop() {if (!run || !isActive(run.status)) return; setPending(true); socket.current?.timeout(12000).emit('stop_run', {run_id: run.id}, (err: Error | null, data: {ok: boolean; error?: string}) => {setPending(false); if (err || !data?.ok) setError(data?.error || 'Stop request failed. Reconnect and try again.');});}
  function newRun() {setPage('workspace'); requestAnimationFrame(() => {form.current?.scrollIntoView({behavior:'smooth', block:'start'}); form.current?.querySelector('select')?.focus({preventScroll:true});});}
  function update<K extends keyof Config>(key: K, value: Config[K]) {setConfig(c => ({...c, [key]: value}));}

  async function inspectHF() {
    const id = config.task_params.dataset_id ?? '';
    if (!id) return;
    setHfInspecting(true); setHfInspect(null);
    try {
      const r = await fetch(`/api/hf_inspect?dataset=${encodeURIComponent(id)}&split=${encodeURIComponent(config.task_params.split || 'train')}`);
      const d = await r.json();
      if (d.error) throw new Error(d.error);
      setHfInspect(d);
      const cols: string[] = d.columns;
      if (cols.length >= 2) update('task_params', {...config.task_params, question_col: config.task_params.question_col || cols[0], answer_col: config.task_params.answer_col || cols[1], reward_type: config.task_params.reward_type || 'contains'});
    } catch(e) {setError(String(e));}
    finally {setHfInspecting(false);}
  }

  async function uploadFile(file: File) {
    setHfUploading(true); setHfInspect(null);
    try {
      const fd = new FormData(); fd.append('file', file);
      const r = await fetch('/api/upload_dataset', {method: 'POST', body: fd});
      const d = await r.json();
      if (d.error) throw new Error(d.error);
      const cols: string[] = d.columns ?? [];
      setHfInspect({columns: cols, sample_rows: d.sample_rows ?? []});
      update('task_params', {...config.task_params, dataset_id: d.path, question_col: cols[0] || '', answer_col: cols[1] || '', reward_type: config.task_params.reward_type || 'contains'});
    } catch(e) {setError(String(e));}
    finally {setHfUploading(false);}
  }

  const isHF = config.task === HF_TASK;
  const isHFCustom = isHF && config.task_params.reward_type === 'custom';
  const canExecute = config.action === 'preview' || config.action === 'demo' || config.allow_unsafe_code || (isHF && !isHFCustom);
  const hfReady = !isHF || (!!config.task_params.dataset_id && (config.action === 'preview' || (!!config.task_params.question_col && !!config.task_params.answer_col && !!config.task_params.reward_type && (config.task_params.reward_type !== 'custom' || !!config.task_params.reward_module))));

  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="/" aria-label="Reinforce home"><div className="brand-mark"><Waypoints size={23}/></div><span>reinforce<span className="brand-period">.</span></span></a>
      <div className="workspace-label"><span className="workspace-avatar">RL</span><div>Local workspace<small>Apple silicon · MLX</small></div><span className="local-dot"/></div>
      <div className="nav-label">WORKSPACE</div>
      <nav aria-label="Main navigation"><button className={page === 'workspace' ? 'nav-item active' : 'nav-item'} onClick={() => setPage('workspace')}><Activity size={17}/>Run workspace<ArrowUpRight size={14}/></button><button className={page === 'history' ? 'nav-item active' : 'nav-item'} onClick={() => setPage('history')}><History size={17}/>Run history<span className="count">{runs.length}</span></button></nav>
      <div className="recent-header"><span className="nav-label">RECENT RUNS</span><button className="icon-button" aria-label="Configure a new run" onClick={newRun}><Plus size={15}/></button></div>
      <div className="recent-runs">{runs.slice(0, 7).map(item => <button key={item.id} onClick={() => watch(item.id)} className={`recent-run ${run?.id === item.id ? 'selected' : ''}`}><span className={`status-dot ${item.status}`}/><div>{item.config.mode === 'demo' ? 'Demo stream' : taskName(item.config.task)}<small>{modelName(item.config.model_id)} <span>· {item.id.slice(0,6)}</span></small></div></button>)}{!runs.length && <div className="empty-recent">Your experiments will<br/>appear here.</div>}</div>
      <div className="sidebar-bottom"><div className="local-notice"><Cpu size={17}/><div>Built to run locally<small>Your compute. Your workspace.</small></div></div><button className="help-button" onClick={() => setHelp(true)}><CircleHelp size={17}/>Workspace guide<ArrowUpRight size={14}/></button></div>
    </aside>
    <div className="main-shell">
      <header className="topbar"><div className="breadcrumb">Workspace <span>/</span> <strong>{page === 'history' ? 'Run history' : 'Run workspace'}</strong></div><div className="topbar-right"><div className={`connection ${connected ? '' : 'offline'}`}>{connected ? <span className="live-dot"/> : <WifiOff size={13}/>} {connected ? 'Backend connected' : 'Reconnecting'}<span className="connection-divider"/><span>LOCAL</span></div><button className="icon-button theme-toggle" onClick={toggleTheme} title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`} aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}>{theme === 'dark' ? <Sun size={16}/> : <Moon size={16}/>}</button></div></header>
      <main>
        <div className="page-heading"><div><div className="eyebrow">EXPERIMENT CONTROL</div><h1>{page === 'history' ? 'Run history' : 'Run workspace'}</h1><p>{page === 'history' ? 'Every experiment, ready to revisit.' : 'From the first token to the final update. Follow it all, live.'}</p></div><div className="heading-actions"><button className="mobile-history" aria-label="Run history" onClick={() => setPage(page === 'history' ? 'workspace' : 'history')}><History size={15}/></button><button className="secondary-button" onClick={newRun}><Plus size={16}/>New run</button></div></div>
        {error && <div className="error-banner" role="alert"><span>{error}</span><button className="icon-button" aria-label="Dismiss error" onClick={() => setError('')}><X size={16}/></button></div>}
        {page === 'history' ? <section className="panel history-panel"><div className="panel-heading"><h2>All experiments</h2><span className="muted">{runs.length} runs</span></div>{runs.length ? <div className="table-scroll"><table><thead><tr><th>Experiment</th><th>Model</th><th>Status</th><th>Steps</th><th>Duration</th><th/></tr></thead><tbody>{runs.map(item => <tr key={item.id}><td><strong>{item.config.mode === 'demo' ? 'Demo stream' : taskName(item.config.task)}</strong><small className="mono">{item.id}</small></td><td>{modelName(item.config.model_id)}</td><td><Status status={item.status}/></td><td className="mono">{item.config.mode === 'demo' || item.config.action === 'train' ? `${item.metric?.step ?? 0} / ${item.config.mode === 'demo' ? Math.min(item.config.steps,12) : item.config.steps}` : '—'}</td><td className="mono">{elapsed(item, clock)}</td><td><button className="text-button" onClick={() => watch(item.id)}>Open <ArrowUpRight size={14}/></button></td></tr>)}</tbody></table></div> : <Empty icon={<History/>} title="A clean slate" detail="Start your first run or try the demo stream to see it here."/>}</section> : <div className="workspace-grid">
          <form ref={form} className="panel configuration" onSubmit={(e: FormEvent) => {e.preventDefault(); launch();}}>
            <div className="panel-heading"><h2><Settings2 size={16}/>Run configuration</h2><span className="mono subtle">01</span></div>
            <div className="config-body">
              <Field label="Task"><div className="select-wrap"><select value={config.task} onChange={e => {const task = e.target.value; const defaults = catalog?.tasks[task]; setConfig(c => ({...c, task, action:'train', ...(defaults ? {steps: defaults.steps, group: defaults.group, lr: defaults.lr} : {}), task_params: {}})); setHfInspect(null); setHfDatasetMode('hf');}}><option value="game_2048.RL2048">2048 strategy</option><option value="mbpp_unittests.MBPPUnitTests">MBPP unit tests</option><option value={HF_TASK}>Custom HF dataset</option></select><ChevronDown size={14}/></div></Field>
              {isHF && <div className="hf-panel">
                <div className="section-label" style={{marginBottom:10}}>DATASET SOURCE</div>
                <div className="model-options hf-source-tabs" role="group">
                  <button type="button" className={hfDatasetMode === 'hf' ? 'chosen' : ''} onClick={() => {setHfDatasetMode('hf'); setHfInspect(null); update('task_params', {});}}>Hugging Face</button>
                  <button type="button" className={hfDatasetMode === 'upload' ? 'chosen' : ''} onClick={() => {setHfDatasetMode('upload'); setHfInspect(null); update('task_params', {});}}>Upload file</button>
                </div>
                {hfDatasetMode === 'hf' ? <>
                  <Field label="Dataset ID"><div className="hf-id-row"><input value={config.task_params.dataset_id ?? ''} placeholder="openai/gsm8k" onChange={e => update('task_params', {...config.task_params, dataset_id: e.target.value})}/><button type="button" className="hf-inspect-btn" onClick={inspectHF} disabled={hfInspecting || !config.task_params.dataset_id}>{hfInspecting ? <LoaderCircle className="spin" size={13}/> : <RotateCcw size={13}/>}Inspect</button></div></Field>
                  <Field label="Training split"><input value={config.task_params.split ?? 'train'} placeholder="train" onChange={e => update('task_params', {...config.task_params, split: e.target.value || 'train'})}/></Field>
                </> : <Field label="File (.csv, .jsonl, .json)"><label className="file-upload-label">{hfUploading ? <><LoaderCircle className="spin" size={13}/>Uploading…</> : 'Choose file…'}<input type="file" accept=".csv,.jsonl,.json" style={{display:'none'}} onChange={e => {const f = e.target.files?.[0]; if (f) uploadFile(f); e.target.value = '';}} disabled={hfUploading}/></label>{config.task_params.dataset_id?.startsWith('/') && <div className="checkpoint" style={{marginTop:8}}><Layers3 size={13}/><span>{config.task_params.dataset_id.split('/').at(-1)}</span></div>}</Field>}
                {hfInspect && <>
                  <div className="field-grid">
                    <Field label="Question column"><div className="select-wrap"><select value={config.task_params.question_col ?? ''} onChange={e => update('task_params', {...config.task_params, question_col: e.target.value})}><option value="">— select —</option>{hfInspect.columns.map(c => <option key={c} value={c}>{c}</option>)}</select><ChevronDown size={14}/></div></Field>
                    <Field label="Answer column"><div className="select-wrap"><select value={config.task_params.answer_col ?? ''} onChange={e => update('task_params', {...config.task_params, answer_col: e.target.value})}><option value="">— select —</option>{hfInspect.columns.map(c => <option key={c} value={c}>{c}</option>)}</select><ChevronDown size={14}/></div></Field>
                  </div>
                  <Field label="Reward function"><div className="select-wrap"><select value={config.task_params.reward_type ?? 'contains'} onChange={e => update('task_params', {...config.task_params, reward_type: e.target.value})}><option value="exact_match">Exact match</option><option value="contains">Contains answer</option><option value="f1_overlap">Token F1</option><option value="regex">Regex pattern</option><option value="custom">Custom function</option></select><ChevronDown size={14}/></div></Field>
                  {config.task_params.reward_type === 'custom' && <Field label="Reward module path (.py)"><input value={config.task_params.reward_module ?? ''} placeholder="/path/to/my_reward.py" onChange={e => update('task_params', {...config.task_params, reward_module: e.target.value})}/></Field>}
                  <Field label="Eval split (optional)"><input value={config.task_params.eval_split ?? ''} placeholder="validation" onChange={e => update('task_params', {...config.task_params, eval_split: e.target.value})}/></Field>
                  <details className="advanced"><summary><span><Layers3 size={13}/>Dataset sample</span><ChevronDown size={14}/></summary><div className="dataset-sample">{hfInspect.sample_rows.slice(0,2).map((row,i) => <div key={i} className="sample-row"><span className="sample-idx mono">{i+1}</span><dl>{Object.entries(row).slice(0,3).map(([k,v]) => <div key={k}><dt>{k}</dt><dd>{String(v).slice(0,120)}</dd></div>)}</dl></div>)}</div></details>
                </>}
              </div>}
              <div className="field"><span id="model-family-label">Model family</span><div className="model-options" role="group" aria-labelledby="model-family-label">{['gemma','qwen','gpt-oss','custom'].map(value => <button type="button" key={value} className={(value === 'custom' ? !['gemma','qwen','gpt-oss'].includes(config.model_id) : config.model_id === value) ? 'chosen' : ''} onClick={() => update('model_id', value === 'custom' ? '' : value)}>{value === 'custom' ? 'Custom' : modelName(value)}</button>)}</div></div>
              {!['gemma','qwen','gpt-oss'].includes(config.model_id) ? <Field label="Checkpoint ID or local path"><input required value={config.model_id} placeholder="mlx-community/model-name" onChange={e => update('model_id',e.target.value)}/></Field> : <div className="checkpoint"><Layers3 size={13}/><span>{catalog?.models[config.model_id]?.replace('mlx-community/','') ?? 'Loading checkpoint…'}</span><span className="bit-badge">MLX</span></div>}
              <Field label="Action"><div className="select-wrap"><select value={config.action} onChange={e => update('action',e.target.value)}><option value="train">Train + evaluate</option><option value="evaluate">Evaluate only</option>{config.task.startsWith('game') ? <option value="demo">2048 board demo</option> : <option value="preview">Preview dataset</option>}</select><ChevronDown size={14}/></div></Field>
              <div className="form-divider"/>
              <div className="section-label">TRAINING PARAMETERS</div>
              <div className="field-grid"><Field label="Steps"><input required type="number" min="1" max="2000" value={config.steps} onChange={e => update('steps',+e.target.value)}/></Field><Field label="Group size"><input required type="number" min="1" max="16" value={config.group} onChange={e => update('group',+e.target.value)}/></Field><Field label="Max tokens"><input required type="number" min="1" max="8192" value={config.max_tokens} onChange={e => update('max_tokens',+e.target.value)}/></Field><Field label="Learning rate"><input required type="number" min="0.000000001" max="1" step="any" value={config.lr} onChange={e => update('lr',+e.target.value)}/></Field></div>
              <Field label="Temperature" detail={config.temp.toFixed(1)}><input className="range" type="range" min="0" max="2" step="0.1" value={config.temp} onChange={e => update('temp',+e.target.value)}/><div className="range-labels"><span>Precise</span><span>Exploratory</span></div></Field>
              <details className="advanced"><summary><span><Settings2 size={13}/>Adapter settings</span><ChevronDown size={14}/></summary><div className="field-grid"><Field label="LoRA rank"><input type="number" min="1" max="128" required value={config.lora_rank} onChange={e => update('lora_rank',+e.target.value)}/></Field><Field label="LoRA layers"><input type="number" min="-1" max="512" required value={config.lora_layers} onChange={e => update('lora_layers',+e.target.value)}/></Field></div><small className="muted">Use −1 for all layers. Zero is invalid.</small></details>
              {['train','evaluate'].includes(config.action) && (!isHF || isHFCustom) && <label className="execution-check"><input type="checkbox" checked={config.allow_unsafe_code} onChange={e => update('allow_unsafe_code',e.target.checked)}/><span>Allow generated Python execution<small>Runs with your user privileges. No sandbox.</small></span></label>}
              <button className="primary-button start-button" type="submit" disabled={!connected || !!active || pending || !canExecute || !hfReady}>{pending ? <LoaderCircle className="spin" size={15}/> : <Play size={15} fill="currentColor"/>}Start run<ArrowRight size={16}/></button>
              <button className="demo-button" type="button" disabled={!connected || !!active || pending} onClick={() => launch(true)}><FlaskConical size={15}/>Try demo stream<ArrowUpRight size={13}/></button><p className="config-caption">Demo streams simulated data.<br/>No model download or code execution.</p>
            </div>
          </form>
          <div className="run-workspace">
            <section className="panel run-overview"><div className="run-title-row"><div className="run-title-icon"><Radio size={20}/></div><div className="run-title"><div><h2>{run ? run.config.mode === 'demo' ? 'Demo stream' : taskName(run.config.task) : 'Ready for your next experiment'}</h2>{run?.config.mode === 'demo' && <span className="demo-badge">SIMULATED</span>}</div><p>{run ? <><span className="mono">{run.id}</span><span> / </span>{modelName(run.config.model_id)}<span> / </span>{label(run.config.action)}</> : 'Configure a run, or explore the live demo.'}</p></div>{run ? <Status status={run.status}/> : <span className="status idle"><span className="status-dot"/>Idle</span>}{run && isActive(run.status) && <button className="stop-button" onClick={stop} disabled={pending || !connected || run.status === 'stopping'}><Square size={11} fill="currentColor"/>{run.status === 'stopping' ? 'Stopping' : 'Stop'}</button>}</div><div className="run-progress"><div><span className={isActive(run?.status) ? 'live-text' : ''}>{isActive(run?.status) && <span className="live-dot"/>}{run?.message ?? 'Waiting for a run'}</span><span className="mono">{elapsed(run,clock)} <span className="muted">elapsed</span></span></div><div className="progress-track"><div style={{width: `${progress}%`}}/></div><div className="progress-label"><span>{tracksSteps ? `${metric?.step ?? 0} of ${total} steps` : run?.status === 'completed' ? 'Task finished' : 'Task progress'}</span><span>{Math.round(progress)}%</span></div></div></section>
            <div className="metrics-grid"><MetricCard icon={<Layers3 size={15}/>} title="TRAINING STEP" value={String(metric?.step ?? '—')} suffix={tracksSteps ? `/ ${total}` : undefined} foot={run?.config.mode === 'demo' ? 'Simulated updates' : 'Completed updates'}/><MetricCard icon={<Activity size={15}/>} title="MEAN REWARD" value={number(metric?.mean_reward)} foot="Across the sample group" accent/><MetricCard icon={<Waypoints size={15}/>} title="LOSS" value={number(metric?.loss)} foot="Sequence objective"/><MetricCard icon={<Zap size={15}/>} title="GENERATION" value={number(latest?.tokens_per_second,1)} suffix="tok/s" foot={latest?.peak_memory_gb ? `${number(latest.peak_memory_gb,2)} GB peak memory` : 'Latest completion'}/></div>
            <section className="panel chart-panel"><div className="panel-heading"><h2>Training trajectory<span className="panel-subtitle">{run?.config.mode === 'demo' ? 'SIMULATED' : 'LIVE METRICS'}</span></h2><div className="segmented"><button className={chart === 'mean_reward' ? 'chosen' : ''} onClick={() => setChart('mean_reward')}>Reward</button><button className={chart === 'loss' ? 'chosen' : ''} onClick={() => setChart('loss')}>Loss</button></div></div><Chart metrics={run?.metrics ?? []} metric={chart}/></section>
            <section className="panel stream-panel"><div className="stream-heading"><div className="tabs" role="tablist" aria-label="Run output">{[{id:'completions',name:'Completions',icon:<Code2 size={15}/>},{id:'logs',name:'Backend logs',icon:<Terminal size={15}/>},{id:'config',name:'Run details',icon:<Settings2 size={15}/>}].map(t => <button key={t.id} role="tab" aria-selected={tab === t.id} className={tab === t.id ? 'selected' : ''} onClick={() => setTab(t.id)}>{t.icon}{t.name}{t.id === 'completions' && !!run?.generations.length && <span className="tab-count">{run.generations.length}</span>}</button>)}</div>{run && <a className="icon-button" href={`/api/runs/${run.id}/events`} title="Download complete event log" aria-label="Download event log"><ArrowDownToLine size={16}/></a>}</div>
              {tab === 'completions' ? <><div className="stream-toolbar"><div className="code-file"><span className="python-dot"/>completion.py{generation && <span className="muted mono">{generation.tokens} tokens</span>}</div><div className="stream-controls">{!!run?.generations.length && <select aria-label="Select completion" value={generationId ?? ''} onChange={e => setGenerationId(e.target.value || null)}><option value="">Follow latest</option>{run.generations.map((g,i) => <option key={g.generation_id} value={g.generation_id}>{g.step ? `Step ${g.step}` : g.phase ?? 'Eval'} · sample {g.sample_index+1} ({i+1})</option>)}</select>}<label className="follow-control"><input type="checkbox" checked={follow} onChange={e => setFollow(e.target.checked)}/>Auto-scroll</label></div></div>{generation ? <><pre className="code-output" ref={output}><code>{generation.text}{!generation.done && isActive(run?.status) && <span className="cursor"/>}</code></pre><div className="stream-footer"><span><span className={`status-dot ${!generation.done && isActive(run?.status) ? 'running' : ''}`}/>{generation.done ? 'Completion received' : isActive(run?.status) ? 'Receiving tokens' : 'Generation interrupted'}<span className="muted">· Sample {generation.sample_index+1} of {generation.group_size}</span></span><span className="mono subtle">SOCKET.IO</span></div><details className="prompt-details"><summary>View source prompt<ChevronDown size={13}/></summary><pre>{generation.prompt}</pre></details></> : <Empty icon={<Code2 size={27}/>} title="Every token, as it happens" detail="Generated completions will stream here directly from the backend." extra={<span className="empty-shortcut"><Radio size={12}/>LIVE TOKEN STREAM</span>}/>}</> : tab === 'logs' ? <><div className="stream-toolbar"><span className="code-file"><Terminal size={14}/>Worker stdout + stderr</span><label className="follow-control"><input type="checkbox" checked={follow} onChange={e => setFollow(e.target.checked)}/>Auto-scroll</label></div>{run?.logs.length ? <pre className="log-output" ref={output}>{run.logs.map(log => <span key={log.seq}>{log.text}</span>)}</pre> : <Empty icon={<Terminal/>} title="Listening to the backend" detail="Worker logs, loading progress, and errors appear here."/>}</> : <div className="run-details">{run ? <><div className="details-heading"><span className="section-label">RUN CONFIGURATION</span><span className="mono subtle">{run.id}</span></div><dl>{Object.entries(run.config).filter(([k]) => k !== 'task_params').map(([key,value]) => <div key={key}><dt>{label(key)}</dt><dd>{String(value)}</dd></div>)}{run.config.task === HF_TASK && Object.entries(run.config.task_params).map(([k,v]) => <div key={k}><dt>{label(k)}</dt><dd className="mono">{String(v)}</dd></div>)}</dl>{run.evaluation != null && <><div className="section-label">LATEST EVALUATION</div><pre>{JSON.stringify(run.evaluation,null,2)}</pre></>}</> : <Empty icon={<Settings2/>} title="Your run, recorded" detail="Settings and evaluation results will be saved with the run."/>}</div>}
            </section><div className="workspace-footnote"><span><span className="status-dot"/>Local compute. Live connection.</span><span>MLX-LM <span className="subtle">/</span> Experimental GRPO</span></div>
          </div>
        </div>}
      </main>
    </div>
    {help && <div className="modal-backdrop" onClick={() => setHelp(false)}><section className="help-modal panel" role="dialog" aria-modal="true" aria-labelledby="help-title" onKeyDown={e => {if (e.key === 'Escape') setHelp(false); if (e.key === 'Tab') {const controls = e.currentTarget.querySelectorAll<HTMLButtonElement>('button'); const first = controls[0], last = controls[controls.length - 1]; if (e.shiftKey && document.activeElement === first) {e.preventDefault(); last.focus();} else if (!e.shiftKey && document.activeElement === last) {e.preventDefault(); first.focus();}}}} onClick={e => e.stopPropagation()}><div className="panel-heading"><h2 id="help-title">Your local run workspace</h2><button className="icon-button" autoFocus aria-label="Close guide" onClick={() => setHelp(false)}><X size={18}/></button></div><div className="help-content"><p>Choose Gemma, Qwen, GPT-OSS, or a local MLX checkpoint, then configure your task and start a run.</p><p>Training and evaluation execute generated Python. Enable the checkbox only when you want code to run with your account's privileges; this workspace has no sandbox.</p><p><strong>Custom HF dataset</strong> lets you load any Hugging Face dataset or upload a CSV/JSONL file. Inspect it to map columns and choose a reward function. No generated code is executed.</p><p><strong>Try demo stream</strong> exercises the real backend connection with simulated tokens and metrics. It does not load models or execute generated code.</p><p>One run uses the local worker at a time. Closing a browser tab does not stop a run. Reopen the workspace to reconnect, or use Stop to terminate the worker and its reward subprocesses.</p><p>Run history and complete event logs are saved locally. Server restarts retain the last saved snapshot; active runs are marked interrupted.</p><button className="primary-button" onClick={() => setHelp(false)}>Got it<Check size={16}/></button></div></section></div>}
  </div>;
}

function Field({label: title, detail, children}: {label: string; detail?: string; children: ReactNode}) {return <label className="field"><span>{title}{detail && <b className="mono">{detail}</b>}</span>{children}</label>;}
function Status({status}: {status: string}) {return <span className={`status ${status}`}><span className={`status-dot ${status}`}/>{label(status)}</span>;}
function Empty({icon,title,detail,extra}: {icon: ReactNode; title: string; detail: string; extra?: ReactNode}) {return <div className="empty-state"><div className="empty-icon">{icon}</div><h3>{title}</h3><p>{detail}</p>{extra}</div>;}
function MetricCard({icon,title,value,suffix,foot,accent}: {icon: ReactNode; title: string; value: string; suffix?: string; foot: string; accent?: boolean}) {return <section className={`metric-card ${accent ? 'accent' : ''}`}><div className="metric-title">{title}{icon}</div><div className="metric-value mono">{value}<span>{suffix}</span></div><p>{foot}</p></section>;}
function Chart({metrics,metric}: {metrics: Metric[]; metric: 'mean_reward' | 'loss'}) {
  const data = metrics.filter(m => Number.isFinite(m[metric]));
  const width = 740, height = 190, left = 42, right = 16, top = 18, bottom = 27;
  const values = data.map(m => m[metric]);
  const min = Math.min(0,...values), max = Math.max(1,...values), span = max-min || 1;
  const lastStep = Math.max(1,data.at(-1)?.step ?? 200);
  const x = (step: number) => left + ((step-1) / Math.max(1,lastStep-1))*(width-left-right);
  const y = (v: number) => top + (1-(v-min)/span)*(height-top-bottom);
  const points = data.map(m => `${x(m.step)},${y(m[metric])}`).join(' ');
  return <div className="chart-wrap"><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${metric === 'loss' ? 'Loss' : 'Mean reward'} by training step`}><defs><linearGradient id="chart-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#b8e59e" stopOpacity=".2"/><stop offset="100%" stopColor="#b8e59e" stopOpacity="0"/></linearGradient></defs>{[0,.25,.5,.75,1].map(v => <g key={v}><line x1={left} x2={width-right} y1={y(min+v*span)} y2={y(min+v*span)} stroke="#2b302d" strokeDasharray="3 5"/><text x={left-12} y={y(min+v*span)+4} textAnchor="end">{(min+v*span).toFixed(2)}</text></g>)}{[0,.25,.5,.75,1].map(v => <text key={v} x={left+v*(width-left-right)} y={height-5} textAnchor="middle">{Math.round(1+v*(lastStep-1))}</text>)}{!!data.length && <><polygon points={`${x(data[0].step)},${height-bottom} ${points} ${x(lastStep)},${height-bottom}`} fill="url(#chart-fill)"/><polyline points={points} fill="none" stroke="#b8e59e" strokeWidth="2.5" strokeLinejoin="round"/>{data.map(m => <circle key={m.step} cx={x(m.step)} cy={y(m[metric])} r={data.length < 30 ? 3 : 1.5} fill="#b8e59e"><title>Step {m.step}: {m[metric].toFixed(4)}</title></circle>)}</>}</svg>{!data.length && <div className="chart-empty"><Activity size={17}/><span>Metrics appear after the first training step</span></div>}<div className="chart-caption"><span><i/>{metric === 'loss' ? 'Sequence loss' : 'Mean group reward'}</span><span>Training step</span></div></div>;
}

createRoot(document.getElementById('root')!).render(<App/>);
