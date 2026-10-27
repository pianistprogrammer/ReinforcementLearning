export type Config = {task: string; model_id: string; action: string; mode: string; steps: number; group: number; lr: number; max_tokens: number; temp: number; lora_rank: number; lora_layers: number; allow_unsafe_code: boolean; task_params: Record<string, string>};
export type RunEvent = {type: string; run_id: string; seq: number; timestamp: string; [key: string]: any};
export type Metric = {step: number; loss: number; mean_reward: number; steps: number};
export type Generation = {generation_id: string; sample_index: number; group_size: number; step?: number; phase?: string; prompt: string; text: string; tokens: number; tokens_per_second?: number; peak_memory_gb?: number; done: boolean};
export type Summary = {id: string; config: Config; status: string; phase: string; message: string; created_at: string; finished_at: string | null; seq: number; metric: Metric | null};
export type Run = Summary & {metrics: Metric[]; logs: RunEvent[]; generations: Generation[]; evaluation: unknown};
export type Catalog = {models: Record<string, string>; tasks: Record<string, {label: string; steps: number; group: number; lr: number; executes_code?: boolean}>};
export type HFInspect = {columns: string[]; sample_rows: Record<string, unknown>[]};

export function applyEvent(run: Run, event: RunEvent): Run {
  if (event.run_id !== run.id || event.seq <= run.seq) return run;
  const next = {...run, seq: event.seq};
  if (event.type === 'phase') {next.phase = event.phase; next.message = event.message;}
  if (event.type === 'status') {next.status = event.status; next.message = event.message; next.finished_at = event.finished_at ?? run.finished_at;}
  if (event.type === 'metric') next.metrics = [...run.metrics, event as unknown as Metric].slice(-2000);
  if (event.type === 'log') next.logs = [...run.logs, event].slice(-300);
  if (event.type === 'evaluation') next.evaluation = event.result;
  if (event.type === 'generation_start') next.generations = [...run.generations, {...event, text: '', tokens: 0, done: false} as unknown as Generation].slice(-12);
  if (event.type === 'token' || event.type === 'generation_end') {
    next.generations = run.generations.map(g => g.generation_id !== event.generation_id ? g : event.type === 'token'
      ? {...g, text: (g.text + event.text).slice(-131072), tokens: event.tokens, tokens_per_second: event.tokens_per_second, peak_memory_gb: event.peak_memory_gb}
      : {...g, done: true});
  }
  return next;
}
