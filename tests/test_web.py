"""Real Socket.IO clients and worker processes, without model downloads."""
import asyncio
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
aiohttp = pytest.importorskip("aiohttp")
socketio = pytest.importorskip("socketio")
from aiohttp.test_utils import TestServer

from rl_framework.web.config import RunConfig
from rl_framework.web.server import create_app, MANAGER


@pytest.mark.parametrize('settings', [
    {}, {'mode': 'training', 'allow_unsafe_code': 'yes'}, {'mode': 'demo', 'steps': 0},
    {'mode': 'demo', 'group': 100}, {'mode': 'demo', 'lr': float('nan')},
    {'mode': 'demo', 'lora_layers': 0}, {'mode': 'demo', 'task': 'os:system'},
    {'mode': 'demo', 'model_id': '\n'}, {'mode': 'demo', 'command': 'arbitrary'},
    {'mode': 'demo', 'steps': True},
])
def test_reject_invalid_and_unapproved_configs(settings):
    with pytest.raises(ValueError):
        RunConfig.parse(settings)


def test_configs_and_argv():
    assert RunConfig.parse({'mode': 'demo'}).allow_unsafe_code is False
    config = RunConfig.parse({'allow_unsafe_code': True, 'model_id': 'qwen'})
    assert '--allow-unsafe-code' in config.argv(Path('/tmp/run'))
    assert '--lora-rank' in config.argv(Path('/tmp/run'))
    from rl_framework.train import parse_args
    assert parse_args(config.argv(Path('/tmp/run'))).model_id == 'qwen'


def test_socket_stream_reconnect_persistence_and_local_origin(tmp_path):
    async def scenario():
        app = create_app(tmp_path)
        async with TestServer(app, host='127.0.0.1') as server:
            url = str(server.make_url('/')).rstrip('/')
            first = socketio.AsyncClient()
            events, snapshots = [], []
            finished = asyncio.Event()

            @first.on('run_event')
            async def event(data):
                events.append(data)
                if data['type'] == 'status' and data['status'] == 'completed':
                    finished.set()

            @first.on('run_snapshot')
            async def snapshot(data):
                snapshots.append(data)

            await first.connect(url, transports=['websocket'])
            try:
                response = await first.call('start_run', {'mode': 'demo', 'steps': 2}, timeout=5)
                assert response['ok']
                run_id = response['run_id']
                await asyncio.wait_for(finished.wait(), 12)
                assert [e['step'] for e in events if e['type'] == 'metric'] == [1, 2]
                assert any(e['type'] == 'token' and e['text'] for e in events)
                assert any(e['type'] == 'log' for e in events)
                assert [e['seq'] for e in events] == sorted({e['seq'] for e in events})
                assert len(snapshots) >= 1
                await first.disconnect()
                await first.connect(url, transports=['websocket'])
                assert (await first.call('watch_run', {'run_id': run_id}))['ok']
                assert snapshots[-1]['status'] == 'completed'
                assert len(snapshots[-1]['generations']) == 2
                assert len(snapshots[-1]['metrics']) == 2
                assert snapshots[-1]['generations'][0]['text'].startswith('def strategy')
                async with aiohttp.ClientSession() as client:
                    async with client.get(url + '/api/runs/' + run_id) as r:
                        assert r.status == 200
                        assert (await r.json())['status'] == 'completed'
                    async with client.get(url + '/api/runs/' + run_id + '/events') as r:
                        saved = [json.loads(line) for line in (await r.text()).splitlines()]
                        assert saved[-1]['status'] == 'completed'
                        assert any(e['type'] == 'token' for e in saved)
                    async with client.get(url + '/api/catalog', headers={'Origin': 'https://other.example'}) as r:
                        assert r.status == 403
                    async with client.get(url + '/api/catalog', headers={'Host': 'rebind.example'}) as r:
                        assert r.status == 403
            finally:
                if first.connected:
                    await first.disconnect()
        restored = create_app(tmp_path)[MANAGER]
        assert restored.runs[run_id].status == 'completed'
        assert len(restored.runs[run_id].metrics) == 2
    asyncio.run(scenario())


def test_single_worker_stop_and_validation_ack(tmp_path):
    async def scenario():
        app = create_app(tmp_path)
        async with TestServer(app, host='127.0.0.1') as server:
            client = socketio.AsyncClient()
            token = asyncio.Event()
            statuses = []

            @client.on('run_event')
            async def event(data):
                if data['type'] == 'token':
                    token.set()
                if data['type'] == 'status':
                    statuses.append(data['status'])

            await client.connect(str(server.make_url('/')).rstrip('/'))
            try:
                assert not (await client.call('start_run', {'model_id': 'gemma'}))['ok']
                response = await client.call('start_run', {'mode': 'demo', 'steps': 12})
                assert response['ok']
                await asyncio.wait_for(token.wait(), 5)
                conflict = await client.call('start_run', {'mode': 'demo'})
                assert not conflict['ok'] and 'active' in conflict['error']
                run = app[MANAGER].runs[response['run_id']]
                assert (await client.call('stop_run', {'run_id': run.id}, timeout=8))['ok']
                assert run.status == 'stopped'
                assert statuses[-1] == 'stopped'
                assert run.process.returncode is not None
                with pytest.raises(ProcessLookupError):
                    os.killpg(run.process.pid, 0)
                assert not (await client.call('watch_run', {'run_id': '../missing'}))['ok']
            finally:
                await client.disconnect()
    asyncio.run(scenario())


def test_worker_failure_reaches_browser(tmp_path):
    async def scenario():
        app = create_app(tmp_path)
        original = asyncio.create_subprocess_exec

        async def failed_process(*args, **kwargs):
            return await original(sys.executable, '-u', '-c',
                                  'import sys; sys.stdin.readline(); print("loader failed",file=sys.stderr); sys.exit(7)', **kwargs)

        async with TestServer(app, host='127.0.0.1') as server:
            client = socketio.AsyncClient()
            finished = asyncio.Event()
            events = []

            @client.on('run_event')
            async def event(data):
                events.append(data)
                if data['type'] == 'status' and data['status'] == 'failed':
                    finished.set()

            await client.connect(str(server.make_url('/')).rstrip('/'))
            try:
                with patch('rl_framework.web.runs.asyncio.create_subprocess_exec', side_effect=failed_process):
                    response = await client.call('start_run', {'mode': 'demo'})
                    await asyncio.wait_for(finished.wait(), 5)
                assert app[MANAGER].runs[response['run_id']].status == 'failed'
                assert any('loader failed' in e.get('text', '') for e in events)
                assert events[-1]['exit_code'] == 7
            finally:
                await client.disconnect()
    asyncio.run(scenario())


def test_restart_marks_incomplete_run_stopped(tmp_path):
    root = tmp_path / 'incomplete'
    root.mkdir()
    (root / 'run.json').write_text(json.dumps({'id': 'incomplete', 'config': {'mode': 'demo'}, 'status': 'running'}))
    manager = create_app(tmp_path)[MANAGER]
    assert manager.runs['incomplete'].status == 'stopped'
    assert 'restart' in manager.runs['incomplete'].message
