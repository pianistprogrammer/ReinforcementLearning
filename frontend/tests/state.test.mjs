import {test} from 'node:test';
import assert from 'node:assert/strict';
import {applyEvent} from '../src/types.ts';

const base = () => ({id:'run-1', seq:0, status:'running', phase:'preparing', message:'', metrics:[], logs:[], generations:[]});
const event = (seq, type, data = {}) => ({run_id:'run-1', seq, type, timestamp:'2026-01-01T00:00:00Z', ...data});

test('reconnect replay cannot duplicate tokens or regress state', () => {
  let run = applyEvent(base(), event(1, 'generation_start', {generation_id:'a', sample_index:0, group_size:1}));
  run = applyEvent(run, event(2, 'token', {generation_id:'a', text:'def ', tokens:1}));
  assert.equal(applyEvent(run, event(2, 'token', {generation_id:'a', text:'def ', tokens:1})), run);
  assert.equal(applyEvent(run, event(1, 'phase', {phase:'loading'})), run);
  assert.equal(applyEvent(run, {...event(9,'status',{status:'failed'}),run_id:'another-run'}), run);
  run = applyEvent(run, event(3,'token',{generation_id:'a',text:'f(): pass',tokens:2}));
  run = applyEvent(run, event(4,'generation_end',{generation_id:'a'}));
  assert.equal(run.generations[0].text, 'def f(): pass');
  assert.equal(run.generations[0].done, true);
});

test('browser retains bounded output and complete latest metrics', () => {
  let run = base();
  let seq = 0;
  for (let i=0;i<15;i++) run=applyEvent(run,event(++seq,'generation_start',{generation_id:String(i)}));
  for (let i=0;i<305;i++) run=applyEvent(run,event(++seq,'log',{text:String(i)}));
  run=applyEvent(run,event(++seq,'metric',{step:1,loss:.4,mean_reward:1,steps:2}));
  run=applyEvent(run,event(++seq,'status',{status:'completed',message:'Run completed',finished_at:'2026-01-01T01:00:00Z'}));
  assert.equal(run.generations.length,12);
  assert.equal(run.generations[0].generation_id,'3');
  assert.equal(run.logs.length,300);
  assert.equal(run.logs[0].text,'5');
  assert.equal(run.metrics[0].loss,.4);
  assert.equal(run.status,'completed');
  assert.equal(run.finished_at,'2026-01-01T01:00:00Z');
});
