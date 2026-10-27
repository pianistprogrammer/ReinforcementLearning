"""Local Socket.IO dashboard: python -m rl_framework.web.server"""
import argparse
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
    if origin and origin != f'{request.scheme}://{request.host}':
        raise web.HTTPForbidden(text='Use the workspace from the same origin')
    return await handler(request)


def create_app(run_dir=None, static_dir=None):
    app = web.Application(middlewares=[local_origin], client_max_size=64 * 1024)
    sio = socketio.AsyncServer(async_mode='aiohttp', max_http_buffer_size=64 * 1024)
    sio.attach(app)
    manager = RunManager(sio, run_dir or Path('artifacts/web-runs').resolve())
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

    static = Path(static_dir) if static_dir else Path(__file__).parent / 'static'

    async def index(request):
        if not (static / 'index.html').is_file():
            return web.Response(status=503, text='Build the frontend first: cd frontend && npm ci && npm run build')
        return web.FileResponse(static / 'index.html', headers={'Cache-Control': 'no-cache'})

    app.router.add_get('/api/catalog', catalog)
    app.router.add_get('/api/runs', runs)
    app.router.add_get('/api/runs/{run_id}', snapshot)
    app.router.add_get('/api/runs/{run_id}/events', events)
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
