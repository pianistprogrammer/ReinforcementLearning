"""Local Socket.IO dashboard: python -m rl_framework.web.server"""
import argparse
import hashlib
import time
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import web
import socketio

from rl_framework.catalog import MODEL_PRESETS
from .config import TASKS
from .runs import RunManager

MANAGER = web.AppKey('run_manager', RunManager)
SOCKET = web.AppKey('socketio', socketio.AsyncServer)


@web.middleware
async def local_origin(request, handler):
    host = urlsplit('//' + request.host).hostname
    if host not in {'localhost', '127.0.0.1', '::1'}:
        raise web.HTTPForbidden(text='This workspace accepts local requests only')
    origin = request.headers.get('Origin')
    if origin and urlsplit(origin).hostname not in {'localhost', '127.0.0.1', '::1'}:
        raise web.HTTPForbidden(text='Use the workspace from the same origin')
    return await handler(request)


def create_app(run_dir=None, static_dir=None):
    resolved_run_dir = (run_dir or Path('artifacts/web-runs')).resolve()
    app = web.Application(middlewares=[local_origin], client_max_size=10 * 1024 * 1024)
    sio = socketio.AsyncServer(async_mode='aiohttp', max_http_buffer_size=64 * 1024,
                               cors_allowed_origins='*')
    sio.attach(app)
    manager = RunManager(sio, resolved_run_dir)
    app[MANAGER], app[SOCKET] = manager, sio

    @sio.event
    async def connect(sid, environ, auth=None):
        await sio.save_session(sid, {})
        await sio.emit('run_list', manager.summaries(), to=sid)

    @sio.on('watch_run')
    async def watch_run(sid, data):
        try:
            await manager.watch(sid, data.get('run_id') if isinstance(data, dict) else None)
            return {'ok': True}
        except ValueError as exc:
            return {'ok': False, 'error': str(exc)}

    @sio.on('start_run')
    async def start_run(sid, data):
        try:
            run = await manager.start(data)
            await manager.watch(sid, run.id)
            return {'ok': True, 'run_id': run.id}
        except (ValueError, TypeError) as exc:
            return {'ok': False, 'error': str(exc)}

    @sio.on('stop_run')
    async def stop_run(sid, data):
        try:
            await manager.stop(data.get('run_id') if isinstance(data, dict) else None)
            return {'ok': True}
        except ValueError as exc:
            return {'ok': False, 'error': str(exc)}

    @sio.on('delete_run')
    async def delete_run(sid, data):
        try:
            await manager.delete(data.get('run_id') if isinstance(data, dict) else None)
            return {'ok': True}
        except ValueError as exc:
            return {'ok': False, 'error': str(exc)}
        except OSError as exc:
            return {'ok': False, 'error': f'Could not remove the run files: {exc}'}

    async def catalog(request):
        return web.json_response({'models': MODEL_PRESETS, 'tasks': TASKS,
                                  'active_run': next((r.id for r in manager.runs.values() if r.status in {'running', 'stopping'}), None)})

    async def runs(request):
        return web.json_response(manager.summaries())

    async def snapshot(request):
        run = manager.runs.get(request.match_info['run_id'])
        if not run:
            raise web.HTTPNotFound()
        return web.json_response(run.snapshot())

    async def events(request):
        run = manager.runs.get(request.match_info['run_id'])
        if not run:
            raise web.HTTPNotFound()
        return web.FileResponse(run.directory / 'events.jsonl', headers={
            'Content-Disposition': f'attachment; filename="{run.id}-events.jsonl"'})

    async def delete_run_http(request):
        try:
            await manager.delete(request.match_info['run_id'])
        except ValueError as exc:
            raise web.HTTPNotFound(text=str(exc))
        except OSError as exc:
            raise web.HTTPInternalServerError(text=str(exc))
        return web.json_response({'ok': True})

    async def hf_inspect(request):
        dataset_id = request.rel_url.query.get('dataset', '').strip()
        split = request.rel_url.query.get('split', 'train').strip() or 'train'
        if not dataset_id:
            return web.json_response({'error': 'dataset parameter required'}, status=400)
        if len(dataset_id) > 512 or any(ord(c) < 32 for c in dataset_id) or dataset_id.startswith('/'):
            return web.json_response({'error': 'Invalid dataset identifier'}, status=400)
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset_id, split=f'{split}[:5]', trust_remote_code=False)

            def _safe(v):
                return str(v)[:200] if not isinstance(v, (str, int, float, bool, type(None))) else v

            rows = [{k: _safe(v) for k, v in dict(row).items()} for row in ds]
            return web.json_response({'columns': list(ds.column_names), 'sample_rows': rows})
        except Exception as exc:
            return web.json_response({'error': str(exc)}, status=400)

    async def upload_dataset(request):
        try:
            reader = await request.multipart()
            part = await reader.next()
            if part is None:
                return web.json_response({'error': 'No file received'}, status=400)
            filename = part.filename or 'upload.csv'
            suffix = Path(filename).suffix.lower()
            if suffix not in {'.csv', '.jsonl', '.json'}:
                return web.json_response({'error': 'Only .csv, .jsonl, and .json files are supported'}, status=400)
            upload_dir = resolved_run_dir / 'uploads'
            upload_dir.mkdir(parents=True, exist_ok=True)
            safe_name = f'{int(time.time())}_{hashlib.md5(filename.encode()).hexdigest()[:8]}{suffix}'
            dest = upload_dir / safe_name
            size = 0
            with open(dest, 'wb') as f:
                while True:
                    chunk = await part.read_chunk()
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > 10 * 1024 * 1024:
                        dest.unlink(missing_ok=True)
                        return web.json_response({'error': 'File exceeds 10 MB limit'}, status=413)
                    f.write(chunk)
            try:
                from datasets import load_dataset

                def _safe(v):
                    return str(v)[:200] if not isinstance(v, (str, int, float, bool, type(None))) else v

                fmt = 'csv' if suffix == '.csv' else 'json'
                ds = load_dataset(fmt, data_files=str(dest), split='train[:5]')
                rows = [{k: _safe(v) for k, v in dict(row).items()} for row in ds]
                columns = list(ds.column_names)
            except Exception:
                columns, rows = [], []
            return web.json_response({'path': str(dest), 'filename': filename,
                                      'columns': columns, 'sample_rows': rows})
        except web.HTTPException:
            raise
        except Exception as exc:
            return web.json_response({'error': str(exc)}, status=500)

    static = Path(static_dir) if static_dir else Path(__file__).parent / 'static'

    async def index(request):
        if not (static / 'index.html').is_file():
            return web.Response(status=503, text='Build the frontend first: cd frontend && npm ci && npm run build')
        return web.FileResponse(static / 'index.html', headers={'Cache-Control': 'no-cache'})

    app.router.add_get('/api/catalog', catalog)
    app.router.add_get('/api/runs', runs)
    app.router.add_get('/api/runs/{run_id}', snapshot)
    app.router.add_get('/api/runs/{run_id}/events', events)
    app.router.add_delete('/api/runs/{run_id}', delete_run_http)
    app.router.add_get('/api/hf_inspect', hf_inspect)
    app.router.add_post('/api/upload_dataset', upload_dataset)
    app.router.add_get('/', index)
    if (static / 'assets').is_dir():
        app.router.add_static('/assets/', static / 'assets')

    async def cleanup(application):
        await manager.close()
        await sio.shutdown()

    app.on_cleanup.append(cleanup)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', choices=['127.0.0.1', 'localhost', '::1'], default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--run-dir', type=Path, default=Path('artifacts/web-runs'))
    args = parser.parse_args()
    web.run_app(create_app(args.run_dir.resolve()), host=args.host, port=args.port, access_log=None)


if __name__ == '__main__':
    main()
