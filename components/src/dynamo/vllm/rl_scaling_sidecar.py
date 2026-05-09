# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""In-process aiohttp HTTP sidecar for the dynamo-vllm worker (RL-Scaling).

This is the missing entrypoint that exposes ``DualModeWorker`` and
``MigrationHandler`` to the rl-scaling-controller. The dynamo runtime's
``serve_endpoint`` is NATS-RPC; the controller's `dual_mode_client` and
`migration_client` speak HTTP. Without this sidecar, ``test-s3.sh`` and
``test-s2.sh`` always fail at the route-probe step.

Routes (default port 9090, override via ``DYNAMO_RL_SIDECAR_PORT``):
  - GET  /healthz              -> {"status":"ok"}
  - GET  /v1/role              -> {"current_role": "decode"|"prefill"}
  - POST /switch_role          -> {"target_role":"decode"|"prefill"}  → DualModeWorker.switch_role
  - POST /migrate_out          -> {"request_id":"…"}                    → MigrationHandler.migrate_out
  - POST /migrate_in           -> {request_id, prompt_tokens, …}        → MigrationHandler.migrate_in
  - GET  /v1/active_requests   -> [...]   debugging aid

The HTTP server runs as an asyncio task in the same event loop as the
worker handler, so all calls share the same AsyncLLM engine without any
IPC. ``InProcessRequestRegistry`` acts as the side-channel between
``BaseWorkerHandler.generate_tokens`` (which records request progress)
and ``EngineRequestTracker`` (which the MigrationHandler reads from).
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- self-pod label
# K8s pod metadata.name is immutable, so a runtime role flip cannot rename
# the pod.  We instead surface the role on a label that operators can view
# with `kubectl get pod -L nvidia.com/dynamo-current-role`.  The patch is
# a strategic-merge JSON to /api/v1/namespaces/<ns>/pods/<name> using the
# in-cluster ServiceAccount token; it is best-effort and silently no-ops
# outside of K8s (e.g. unit tests, local dev).
_SA_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
_SA_CA_PATH    = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
_SA_NS_PATH    = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"


async def _patch_self_pod_label(label_key: str, label_value: str) -> None:
    """Patch a single label on the current pod via the in-cluster K8s API.

    No-ops gracefully if any of the SA files / env vars are missing, so
    the same code runs unmodified in unit tests and bare-metal dev.
    """
    pod_name = os.environ.get("POD_NAME") or os.environ.get("HOSTNAME")
    if not pod_name:
        return
    if not (
        os.path.isfile(_SA_TOKEN_PATH)
        and os.path.isfile(_SA_NS_PATH)
    ):
        return  # not running in a K8s pod with a mounted SA

    api_host = os.environ.get("KUBERNETES_SERVICE_HOST")
    api_port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
    if not api_host:
        return

    with open(_SA_TOKEN_PATH, "r", encoding="utf-8") as fh:
        token = fh.read().strip()
    with open(_SA_NS_PATH, "r", encoding="utf-8") as fh:
        namespace = fh.read().strip()

    url = (
        f"https://{api_host}:{api_port}/api/v1/namespaces/"
        f"{namespace}/pods/{pod_name}"
    )
    # Strategic-merge patch: only the listed label is changed; everything
    # else on the pod is left untouched.  Escape '/' as '~1' per JSON-Patch
    # rules — but strategic-merge takes a plain map, so a nested dict works.
    patch_body = {"metadata": {"labels": {label_key: label_value}}}

    from aiohttp import ClientSession, ClientTimeout, TCPConnector  # noqa: WPS433

    ssl_ctx = None
    if os.path.isfile(_SA_CA_PATH):
        import ssl  # noqa: WPS433
        ssl_ctx = ssl.create_default_context(cafile=_SA_CA_PATH)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/strategic-merge-patch+json",
        "Accept":        "application/json",
    }
    timeout = ClientTimeout(total=5)
    connector = TCPConnector(ssl=ssl_ctx) if ssl_ctx else None
    async with ClientSession(timeout=timeout, connector=connector) as sess:
        async with sess.patch(url, json=patch_body, headers=headers) as resp:
            if resp.status >= 300:
                txt = await resp.text()
                logger.warning(
                    "[RLScalingSidecar] pod label patch returned HTTP %d: %s",
                    resp.status, txt[:200],
                )
            else:
                logger.info(
                    "[RLScalingSidecar] patched pod label %s=%s on %s/%s",
                    label_key, label_value, namespace, pod_name,
                )


# --------------------------------------------------------------------- registry
@dataclass
class _RequestSnapshot:
    """In-flight state that ``MigrationHandler.migrate_out`` needs."""

    request_id: str
    prompt_tokens: list[int]
    sampling_params_dict: dict
    stop_conditions: dict = field(default_factory=dict)
    generated_tokens: list[int] = field(default_factory=list)


class InProcessRequestRegistry:
    """Tracks in-flight requests on the local worker.

    Kept deliberately tiny: the only writers are ``register`` /
    ``record_tokens`` / ``deregister`` (called from the handler's generation
    loop), and the only readers are ``EngineRequestTracker`` methods.

    Thread-safety: the worker is single-threaded asyncio, so plain dict
    mutations are safe.
    """

    def __init__(self) -> None:
        self._snapshots: dict[str, _RequestSnapshot] = {}

    # ------------------------------------------------------------ writers
    def register(
        self,
        request_id: str,
        prompt_tokens: Iterable[int],
        sampling_params_dict: dict,
        stop_conditions: Optional[dict] = None,
    ) -> None:
        self._snapshots[request_id] = _RequestSnapshot(
            request_id=request_id,
            prompt_tokens=list(prompt_tokens),
            sampling_params_dict=dict(sampling_params_dict or {}),
            stop_conditions=dict(stop_conditions or {}),
        )

    def record_tokens(self, request_id: str, new_token_ids: Iterable[int]) -> None:
        snap = self._snapshots.get(request_id)
        if snap is None:
            return
        snap.generated_tokens.extend(int(t) for t in new_token_ids)

    def deregister(self, request_id: str) -> None:
        self._snapshots.pop(request_id, None)

    # ------------------------------------------------------------ readers
    def get(self, request_id: str) -> Optional[_RequestSnapshot]:
        return self._snapshots.get(request_id)

    def active_ids(self) -> list[str]:
        return list(self._snapshots.keys())


# --------------------------------------------------------------------- tracker
class EngineRequestTracker:
    """Implements the ``RequestTracker`` Protocol from migration.py.

    Backed by:
      - ``InProcessRequestRegistry`` for state lookups + active id listing.
      - ``AsyncLLM.abort`` for the abort path.
      - ``submit_request_callback`` for resubmission. Resubmission
        re-injects the request into the engine via ``AsyncLLM.generate``,
        but the streamed tokens are **discarded**: the original client
        connection terminated with the abort, and re-attaching it is a
        frontend-side responsibility (tracked separately as Phase-2.C).
        The migration's job ends when the new engine starts producing
        tokens; downstream metrics still observe the per-token throughput.
    """

    def __init__(
        self,
        engine_client: Any,
        registry: InProcessRequestRegistry,
        submit_request_callback: Callable[[str, dict], Awaitable[None]],
    ) -> None:
        self._engine = engine_client
        self._registry = registry
        self._submit = submit_request_callback

    async def get_request_state(self, request_id: str) -> Optional[dict]:
        snap = self._registry.get(request_id)
        if snap is None:
            return None
        return {
            "prompt_tokens": list(snap.prompt_tokens),
            "generated_tokens": list(snap.generated_tokens),
            "sampling_params": dict(snap.sampling_params_dict),
            "stop_conditions": dict(snap.stop_conditions),
        }

    async def abort_request(self, request_id: str) -> None:
        try:
            await self._engine.abort(request_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Sidecar] engine.abort(%s) raised: %s", request_id, exc)
        finally:
            # Even if abort raised, the request is logically gone from this
            # worker's perspective (the source-of-truth is the registry).
            self._registry.deregister(request_id)

    async def submit_request(self, request_id: str, payload: dict) -> None:
        await self._submit(request_id, payload)

    async def list_active_request_ids(self) -> Iterable[str]:
        return self._registry.active_ids()


# --------------------------------------------------------------------- HTTP app
def _try_import_aiohttp():
    try:
        from aiohttp import web  # noqa: WPS433
        return web
    except ImportError as exc:
        raise ImportError(
            "RL-Scaling sidecar requires aiohttp. Install with: pip install aiohttp"
        ) from exc


def build_app(
    *,
    dual_mode_worker: Optional[Any] = None,
    migration_handler: Optional[Any] = None,
    initial_role: Optional[str] = None,
    registry: Optional[InProcessRequestRegistry] = None,
):
    """Build the aiohttp ``Application`` and register routes.

    All four collaborators are optional so callers can mount partial
    surfaces (e.g. a prefill-only worker doesn't need migration). Missing
    collaborators just turn the corresponding routes into 503s.
    """
    web = _try_import_aiohttp()

    async def healthz(_request):
        return web.json_response({"status": "ok"})

    async def get_role(_request):
        if dual_mode_worker is None:
            return web.json_response(
                {"current_role": initial_role or "unknown"}
            )
        return web.json_response({"current_role": dual_mode_worker.current_role})

    async def post_switch_role(request):
        if dual_mode_worker is None:
            return web.json_response(
                {"status": "error", "message": "dual_mode not enabled on this worker"},
                status=503,
            )
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"status": "error", "message": "invalid JSON"}, status=400)
        target = body.get("target_role")
        if not isinstance(target, str):
            return web.json_response(
                {"status": "error", "message": "target_role must be a string"},
                status=400,
            )
        result = await dual_mode_worker.switch_role(target)
        # DualModeWorker returns {status, new_role, switch_time_ms, ...}
        # Best-effort: surface the new role as a pod label so that
        # `kubectl get pod -L nvidia.com/dynamo-current-role` reflects
        # the runtime state (pod metadata.name is immutable in K8s).
        if isinstance(result, dict) and result.get("status") == "ok":
            new_role = result.get("new_role") or target
            try:
                await _patch_self_pod_label(
                    "nvidia.com/dynamo-current-role", str(new_role)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[RLScalingSidecar] could not patch self pod label "
                    "(non-fatal): %s", exc
                )
        return web.json_response(result)

    async def post_migrate_out(request):
        if migration_handler is None:
            return web.json_response(
                {"status": "error", "message": "migration not enabled on this worker"},
                status=503,
            )
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"status": "error", "message": "invalid JSON"}, status=400)
        result = await migration_handler.migrate_out(body)
        return web.json_response(result)

    async def post_migrate_in(request):
        if migration_handler is None:
            return web.json_response(
                {"status": "error", "message": "migration not enabled on this worker"},
                status=503,
            )
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"status": "error", "message": "invalid JSON"}, status=400)
        result = await migration_handler.migrate_in(body)
        return web.json_response(result)

    async def get_active(_request):
        if registry is None:
            return web.json_response([])
        return web.json_response(registry.active_ids())

    app = web.Application()
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/v1/role", get_role)
    app.router.add_post("/switch_role", post_switch_role)
    app.router.add_post("/migrate_out", post_migrate_out)
    app.router.add_post("/migrate_in", post_migrate_in)
    app.router.add_get("/v1/active_requests", get_active)
    return app


async def start_sidecar(
    *,
    dual_mode_worker: Optional[Any] = None,
    migration_handler: Optional[Any] = None,
    initial_role: Optional[str] = None,
    registry: Optional[InProcessRequestRegistry] = None,
    host: str = "0.0.0.0",
    port: Optional[int] = None,
):
    """Start the sidecar HTTP server. Returns ``(runner, site)``.

    Caller is responsible for ``await runner.cleanup()`` on shutdown.
    """
    web = _try_import_aiohttp()
    if port is None:
        port = int(os.environ.get("DYNAMO_RL_SIDECAR_PORT", "9091"))
    app = build_app(
        dual_mode_worker=dual_mode_worker,
        migration_handler=migration_handler,
        initial_role=initial_role,
        registry=registry,
    )
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info(
        "[RLScalingSidecar] listening on %s:%d (dual_mode=%s, migration=%s)",
        host,
        port,
        bool(dual_mode_worker),
        bool(migration_handler),
    )
    return runner, site


# ------------------------------------------------------------- nixl meta helper
def make_nixl_meta_provider(vllm_config: Any) -> Callable[[], Optional[dict]]:
    """Build a ``nixl_meta_provider`` callback for ``MigrationHandler``.

    Reads ``engine_id``, ``nixl_side_channel_host``, ``nixl_side_channel_port``
    from the engine's ``KvTransferConfig`` *every* time it's invoked. Reading
    lazily lets the worker swap the underlying connector during a role flip
    without re-wiring the migration handler.

    Returns ``None`` if KV transfer is not configured (e.g. legacy
    deployments without disagg-PD).
    """

    def _provider() -> Optional[dict]:
        try:
            kv_cfg = getattr(vllm_config, "kv_transfer_config", None)
            if kv_cfg is None:
                return None
            engine_id = getattr(kv_cfg, "engine_id", None)
            host = getattr(kv_cfg, "nixl_side_channel_host", None)
            port = getattr(kv_cfg, "nixl_side_channel_port", None)
            if not (engine_id and host and port):
                return None
            return {"engine_id": str(engine_id), "host": str(host), "port": int(port)}
        except Exception:  # noqa: BLE001
            logger.debug("nixl_meta_provider: failed to read kv_transfer_config", exc_info=True)
            return None

    return _provider


# --------------------------------------------------------- submit_request impl
def make_submit_request_callback(engine_client: Any) -> Callable[[str, dict], Awaitable[None]]:
    """Build a submit_request callback bound to an AsyncLLM client.

    The callback re-injects a migration's payload into the engine via
    ``engine.generate(...)``. The streaming tokens are intentionally
    drained-and-discarded inside a fire-and-forget asyncio task because
    the original client connection (which terminated when migrate_out
    aborted) cannot be re-attached server-side; that is a frontend-side
    concern (Phase-2.C). The drain ensures the engine's state machine
    progresses normally.
    """
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import SamplingParams

    async def _submit(request_id: str, payload: dict) -> None:
        prompt_tokens = list(payload.get("prompt_tokens") or [])
        if not prompt_tokens:
            raise ValueError(f"submit_request[{request_id}]: empty prompt_tokens")
        sp_dict = payload.get("sampling_params") or {}
        # Build SamplingParams without exploding on unknown fields. We only
        # forward the well-defined subset; anything unrecognised is ignored.
        sp_kwargs = {}
        for key in ("temperature", "top_p", "top_k", "max_tokens", "min_tokens",
                    "presence_penalty", "frequency_penalty", "repetition_penalty",
                    "stop", "stop_token_ids", "seed", "n"):
            if key in sp_dict:
                sp_kwargs[key] = sp_dict[key]
        sampling_params = SamplingParams(**sp_kwargs)
        kv_params = payload.get("kv_transfer_params")
        if kv_params:
            if sampling_params.extra_args is None:
                sampling_params.extra_args = {}
            sampling_params.extra_args["kv_transfer_params"] = dict(kv_params)

        prompt = TokensPrompt(prompt_token_ids=prompt_tokens)

        async def _drain():
            try:
                async for _ in engine_client.generate(prompt, sampling_params, request_id):
                    pass
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[Sidecar] resubmitted request %s drain raised: %s",
                    request_id,
                    exc,
                )

        asyncio.create_task(_drain())

    return _submit
