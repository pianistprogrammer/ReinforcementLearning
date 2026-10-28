"""One local training worker at a time, with durable events and reconnect snapshots."""
import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import sys
from uuid import uuid4

from .config import RunConfig

ACTIVE = {'running', 'stopping'}


def now():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Run:
    id: str
    config: dict
    directory: Path
    status: str = 'running'
    phase: str = 'preparing'
    message: str = 'Starting worker'
    step: int = 0
    created_at: str = field(default_factory=now)
    finished_at: str | None = None
    seq: int = 0
    metrics: list = field(default_factory=list)
    logs: deque = field(default_factory=lambda: deque(maxlen=300))
    generations: list = field(default_factory=list)
    evaluation: object = None
    process: object = None
    task: object = None
    stop_requested: bool = False
    event_file: object = None

    def summary(self):
        return {'id': self.id, 'config': self.config, 'status': self.status, 'phase': self.phase,
                'message': self.message, 'step': self.step, 'created_at': self.created_at,
                'finished_at': self.finished_at,
                'seq': self.seq, 'metric': self.metrics[-1] if self.metrics else None}

    def snapshot(self):
        return {**self.summary(), 'metrics': self.metrics, 'logs': list(self.logs),
                'generations': self.generations, 'evaluation': self.evaluation}

    def save(self):
        path = self.directory / 'run.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.snapshot(), allow_nan=False))
        temp.replace(path)

    def apply(self, event):
        kind = event['type']
        if kind == 'phase':
            self.phase, self.message = event['phase'], event.get('message', event['phase'])
            if isinstance(event.get('step'), int):
                self.step = event['step']
        elif kind == 'status':
            self.status = event['status']
            self.finished_at = event.get('finished_at', self.finished_at)
            self.message = event.get('message', self.message)
        elif kind == 'metric':
            self.metrics.append(event)
            self.metrics = self.metrics[-2000:]
        elif kind == 'log':
            self.logs.append(event)
        elif kind == 'generation_start':
            self.generations.append({**event, 'text': '', 'tokens': 0, 'done': False})
            self.generations = self.generations[-12:]
        elif kind in ('token', 'generation_end'):
            generation = next((g for g in self.generations if g['generation_id'] == event['generation_id']), None)
            if generation is not None:
                if kind == 'token':
                    generation['text'] = (generation['text'] + event.get('text', ''))[-131072:]
                    generation['tokens'] = event['tokens']
                    generation['tokens_per_second'] = event.get('tokens_per_second')
                    generation['peak_memory_gb'] = event.get('peak_memory_gb')
                else:
                    generation['done'] = True
        elif kind == 'evaluation':
            self.evaluation = event.get('result')


class RunManager:
    def __init__(self, sio, root):
        self.sio, self.root = sio, Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs = {}
        self.lifecycle_lock = asyncio.Lock()
        self.publish_lock = asyncio.Lock()
        self.load()

    def load(self):
        for path in sorted(self.root.glob('*/run.json')):
            try:
                data = json.loads(path.read_text())
                run = Run(data['id'], data['config'], path.parent)
                for key in ('status', 'phase', 'message', 'step', 'created_at', 'finished_at', 'seq',
                            'metrics', 'generations', 'evaluation'):
                    if key in data:
                        setattr(run, key, data[key])
                run.logs.extend(data.get('logs', []))
                if run.status in ACTIVE:
                    run.status, run.message, run.finished_at = 'stopped', 'Interrupted by server restart', now()
                    run.save()
                self.runs[run.id] = run
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def summaries(self):
        return [run.summary() for run in sorted(self.runs.values(), key=lambda r: r.created_at, reverse=True)]

    async def publish(self, run, data):
        async with self.publish_lock:
            run.seq += 1
            event = {**data, 'run_id': run.id, 'seq': run.seq, 'timestamp': now()}
            run.apply(event)
            if run.event_file:
                run.event_file.write(json.dumps(event, allow_nan=False) + '\n')
            if event['type'] in ('phase', 'status', 'metric', 'evaluation'):
                run.save()
            await self.sio.emit('run_event', event, room=run.id)
            if event['type'] in ('phase', 'status', 'metric'):
                await self.sio.emit('run_list', self.summaries())

    async def watch(self, sid, run_id):
        if not isinstance(run_id, str) or run_id not in self.runs:
            raise ValueError('Run not found')
        async with self.publish_lock:
            session = await self.sio.get_session(sid)
            if session.get('run_id'):
                await self.sio.leave_room(sid, session['run_id'])
            await self.sio.enter_room(sid, run_id)
            await self.sio.save_session(sid, {'run_id': run_id})
            await self.sio.emit('run_snapshot', self.runs[run_id].snapshot(), to=sid)

    async def start(self, data):
        config = RunConfig.parse(data)
        async with self.lifecycle_lock:
            if any(run.status in ACTIVE for run in self.runs.values()):
                raise ValueError('Another run is active. Stop it before starting a new run.')
            run_id = uuid4().hex[:12]
            directory = self.root / run_id
            directory.mkdir()
            run = Run(run_id, config.to_dict(), directory)
            run.event_file = (directory / 'events.jsonl').open('a', buffering=1)
            self.runs[run_id] = run
            run.save()
            run.task = asyncio.create_task(self.execute(run), name=f'run-{run.id}')
            await self.sio.emit('run_list', self.summaries())
            return run

    def signal_group(self, run, sig):
        if run.process:
            try:
                os.killpg(run.process.pid, sig)
            except ProcessLookupError:
                pass

    async def stop(self, run_id):
        if not isinstance(run_id, str) or run_id not in self.runs:
            raise ValueError('Run not found')
        run = self.runs[run_id]
        if run.status not in ACTIVE:
            return
        run.stop_requested = True
        await self.publish(run, {'type': 'status', 'status': 'stopping', 'message': 'Stopping worker…'})
        self.signal_group(run, signal.SIGTERM)
        if run.task:
            try:
                await asyncio.wait_for(asyncio.shield(run.task), timeout=4)
            except asyncio.TimeoutError:
                self.signal_group(run, signal.SIGKILL)
                await run.task

    async def execute(self, run):
        try:
            source = Path(__file__).resolve().parents[2]
            env = dict(os.environ, PYTHONUNBUFFERED='1')
            entries = [str(source), *env.get('PYTHONPATH', '').split(os.pathsep)]
            env['PYTHONPATH'] = os.pathsep.join(e for e in entries if e)
            # Workers must not inherit the server's launch directory: it can be a
            # deleted temp dir, which makes CPython fail at startup (getpath).
            run.process = await asyncio.create_subprocess_exec(
                sys.executable, '-u', '-m', 'rl_framework.web.worker',
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, start_new_session=True, env=env,
                cwd=str(source), limit=1024 * 1024)
            run.process.stdin.write(json.dumps({'config': run.config, 'output_dir': str(run.directory / 'artifacts')}).encode() + b'\n')
            await run.process.stdin.drain()
            run.process.stdin.close()
            if run.stop_requested:
                self.signal_group(run, signal.SIGTERM)

            async def stdout():
                while line := await run.process.stdout.readline():
                    text = line.decode(errors='replace')
                    if text.startswith('@rl:'):
                        try:
                            event = json.loads(text[4:])
                            if not isinstance(event, dict) or not isinstance(event.get('type'), str):
                                raise ValueError('Invalid worker event')
                            await self.publish(run, event)
                            continue
                        except (ValueError, TypeError):
                            pass
                    await self.publish(run, {'type': 'log', 'stream': 'stdout', 'text': text[:8192]})

            async def stderr():
                while chunk := await run.process.stderr.read(4096):
                    await self.publish(run, {'type': 'log', 'stream': 'stderr', 'text': chunk.decode(errors='replace')})

            await asyncio.gather(stdout(), stderr())
            code = await run.process.wait()
            status = 'stopped' if run.stop_requested else ('completed' if code == 0 else 'failed')
            message = {'stopped': 'Run stopped', 'completed': 'Run completed', 'failed': f'Worker exited with code {code}'}[status]
            await self.publish(run, {'type': 'status', 'status': status, 'message': message,
                                     'finished_at': now(), 'exit_code': code})
        except Exception as exc:
            self.signal_group(run, signal.SIGKILL)
            if run.process:
                await run.process.wait()
            await self.publish(run, {'type': 'log', 'stream': 'stderr', 'text': f'{type(exc).__name__}: {exc}\n'})
            await self.publish(run, {'type': 'status', 'status': 'stopped' if run.stop_requested else 'failed',
                                     'message': str(exc), 'finished_at': now()})
        finally:
            # Also remove any reward subprocess still in the worker's process group.
            self.signal_group(run, signal.SIGKILL)
            if run.event_file:
                run.event_file.close()
                run.event_file = None

    async def delete(self, run_id):
        if not isinstance(run_id, str) or run_id not in self.runs:
            raise ValueError('Run not found')
        run = self.runs[run_id]
        if run.status in ACTIVE:
            await self.stop(run_id)
        directory = Path(run.directory).resolve()
        if directory.parent != self.root.resolve():
            raise ValueError('Refusing to delete a run outside the run directory')
        async with self.lifecycle_lock:
            self.runs.pop(run_id, None)
            if run.event_file:
                run.event_file.close()
                run.event_file = None
            await asyncio.to_thread(shutil.rmtree, directory, True)
        await self.sio.emit('run_list', self.summaries())
        await self.sio.emit('run_deleted', {'run_id': run_id})

    async def close(self):
        await asyncio.gather(*(self.stop(run.id) for run in self.runs.values() if run.status in ACTIVE))
