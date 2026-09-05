"""Real, validating replay backend for Hermes — plus the guards that make a
broken backend fail loudly instead of scoring zero.

Two problems this module exists to solve.

**Silent failure (P1).** ``skillopt_sleep.backend.CliBackend._call`` catches every
exception and never inspects the exit code, so it returns ``""`` when the CLI is
unauthenticated, times out, or is missing. An empty attempt then scores 0.0 and
an unparseable judge returns ``judge-parse-failed`` — also 0.0. The run exits 0
and stages a plausible report. A dead backend is indistinguishable from one that
legitimately found no improvement, which makes every measurement untrustworthy.
``StrictBackend`` turns those empties into ``BackendCallError``.

**No usable transport (P2).** The ``claude`` CLI needs its own login and the
``codex`` CLI exceeds the per-call timeout. Hermes already has authenticated
providers, so ``HermesLlmBackend`` routes through ``agent.auxiliary_client``
instead — inheriting ``CliBackend``'s causal ``attempt()`` (the skill goes into
the prompt), its rubric ``judge()``, and its ``(task, skill, memory)`` cache,
and overriding only the transport.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Optional, Tuple

from skillopt_sleep.backend import CliBackend
from skillopt_sleep.types import TaskRecord

logger = logging.getLogger(__name__)

#: Pinned rather than left to Hermes's routing: an unpinned auxiliary call falls
#: through the provider chain and can return ``None`` content when an upstream
#: provider has a credit or token-exchange fault, which reads as a silent empty.
DEFAULT_PROVIDER = os.environ.get("SKILLOPT_HERMES_PROVIDER", "deepseek")
DEFAULT_MODEL = os.environ.get("SKILLOPT_HERMES_MODEL", "deepseek-v4-flash")

_PROBE_PROMPT = "Reply with exactly: PONG"


class BackendCallError(RuntimeError):
    """A backend call failed or produced nothing usable.

    Deliberately fatal to the run: a proposal built on missing replays must
    never reach staging.
    """


# ── Hermes environment ───────────────────────────────────────────────────────

def load_hermes_env(hermes_home: str = "") -> int:
    """Load ``<hermes_home>/.env`` into ``os.environ``; return the count applied.

    Hermes's own entrypoints load this file, so provider credentials live there
    rather than in the ambient environment. Without it an auxiliary call falls
    through every provider and returns empty content instead of raising.
    Existing variables win, so an explicit export still overrides the file.
    """
    from hermes_skillopt.sleep import _resolve_hermes_home

    env_path = os.path.join(hermes_home or _resolve_hermes_home(), ".env")
    if not os.path.isfile(env_path):
        return 0
    applied = 0
    try:
        with open(env_path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                if not key or key in os.environ:
                    continue
                os.environ[key] = value.strip().strip('"').strip("'")
                applied += 1
    except OSError as exc:
        logger.debug("Could not read %s: %s", env_path, exc)
    return applied


def _resolve_agent_path(agent_path: str = "", hermes_home: str = "") -> str:
    """Directory of the hermes-agent checkout that provides ``agent.auxiliary_client``."""
    from hermes_skillopt.sleep import _resolve_hermes_home

    if agent_path:
        return agent_path
    env = os.environ.get("HERMES_AGENT_PATH", "")
    if env:
        return env
    return os.path.join(hermes_home or _resolve_hermes_home(), "hermes-agent")


def _import_call_llm(agent_path: str):
    """Import Hermes's auxiliary LLM entrypoint, or explain precisely why not."""
    if agent_path and os.path.isdir(agent_path) and agent_path not in sys.path:
        sys.path.insert(0, agent_path)
    try:
        from agent.auxiliary_client import call_llm
    except Exception as exc:  # ImportError, and anything the package raises on import
        raise BackendCallError(
            f"Could not import agent.auxiliary_client from {agent_path!r} ({exc}). "
            "Point --hermes-agent-path (or HERMES_AGENT_PATH) at a hermes-agent checkout."
        ) from exc
    return call_llm


# ── The backend ──────────────────────────────────────────────────────────────

class HermesLlmBackend(CliBackend):
    """Replay through Hermes's configured providers.

    Only the transport differs from :class:`CliBackend`; the causal attempt
    prompt, the rubric judge, the reflect prompt and the response cache are all
    inherited, so this stays correct as upstream evolves them.
    """

    name = "hermes"

    def __init__(
        self,
        model: str = "",
        *,
        provider: str = "",
        hermes_home: str = "",
        agent_path: str = "",
        timeout: int = 180,
    ) -> None:
        super().__init__(model=model or DEFAULT_MODEL, timeout=timeout)
        self.provider = provider or DEFAULT_PROVIDER
        self.hermes_home = hermes_home
        self.agent_path = _resolve_agent_path(agent_path, hermes_home)
        self._call_llm = None
        self._real_tokens = 0
        load_hermes_env(hermes_home)

    def _ensure_client(self):
        if self._call_llm is None:
            self._call_llm = _import_call_llm(self.agent_path)
        return self._call_llm

    def _call(self, prompt: str, *, max_tokens: int = 1024) -> str:
        call_llm = self._ensure_client()
        try:
            response = call_llm(
                task="skillopt",
                provider=self.provider or None,
                model=self.model or None,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0,
                timeout=self.timeout,
            )
        except Exception as exc:
            raise BackendCallError(f"Hermes auxiliary call failed: {type(exc).__name__}: {exc}") from exc

        content = _extract_content(response)
        # A provider fault (expired credit, degraded token exchange) surfaces as
        # None content rather than an exception — the exact silent-zero this
        # backend exists to avoid.
        if not (content or "").strip():
            raise BackendCallError(
                f"Hermes auxiliary call returned no content "
                f"(provider={self.provider or 'auto'}, model={self.model or 'auto'}). "
                "Check credentials in <hermes-home>/.env and provider credit."
            )
        self._real_tokens += _usage_tokens(response)
        return content

    def tokens_used(self) -> int:
        """Provider-reported tokens, falling back to the inherited estimate."""
        return self._real_tokens or super().tokens_used()

    def probe(self) -> str:
        """One cheap round-trip so a dead backend fails in seconds, not mid-run."""
        return self._call(_PROBE_PROMPT, max_tokens=16)


def _extract_content(response: Any) -> str:
    """Pull assistant text out of an OpenAI-shaped response, tolerantly."""
    try:
        choice = response.choices[0]
    except (AttributeError, IndexError, TypeError):
        return ""
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if content is None and isinstance(choice, dict):
        content = (choice.get("message") or {}).get("content")
    return content or ""


def _usage_tokens(response: Any) -> int:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    total = getattr(usage, "total_tokens", None)
    if isinstance(total, int):
        return total
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    return int(prompt) + int(completion)


# ── Loud-failure guard ───────────────────────────────────────────────────────

class StrictBackend:
    """Wrap any backend so empty or unparseable results raise instead of scoring 0.

    Delegates every operation to ``inner``. The guard sits on ``_call`` where one
    exists, which is the single point every attempt, judge and reflect prompt
    passes through — so it covers the CLI backends without forking upstream, and
    keeps working when upstream changes its prompts.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.name = getattr(inner, "name", "backend")
        original = getattr(inner, "_call", None)
        if callable(original):
            inner._call = self._guard(original)

    def _guard(self, original):
        backend_name = self.name

        def guarded(prompt: str, *, max_tokens: int = 1024) -> str:
            out = original(prompt, max_tokens=max_tokens)
            if not (out or "").strip():
                raise BackendCallError(
                    f"Backend {backend_name!r} returned an empty response. "
                    "Upstream swallows the cause, so check the tool directly: an "
                    "unauthenticated CLI, a per-call timeout, or a provider fault "
                    "all present this way."
                )
            return out

        return guarded

    # -- delegated operations ------------------------------------------------

    def attempt(self, task: TaskRecord, skill: str, memory: str) -> str:
        response = self._inner.attempt(task, skill, memory)
        if not (response or "").strip():
            raise BackendCallError(
                f"Backend {self.name!r} produced no response for task {task.id}; "
                "refusing to score it as a failure."
            )
        return response

    def judge(self, task: TaskRecord, response: str) -> Tuple[float, float, str]:
        hard, soft, reason = self._inner.judge(task, response)
        if reason == "judge-parse-failed":
            raise BackendCallError(
                f"Judge returned unparseable output for task {task.id}. Scoring it "
                "0.0 would be indistinguishable from a genuinely bad answer."
            )
        return hard, soft, reason

    def probe(self) -> Optional[str]:
        probe = getattr(self._inner, "probe", None)
        if callable(probe):
            return probe()
        call = getattr(self._inner, "_call", None)
        return call(_PROBE_PROMPT, max_tokens=16) if callable(call) else None

    def __getattr__(self, item: str) -> Any:
        # reflect, tokens_used, preferences, attempt_with_tools, ...
        return getattr(self._inner, item)


def build_validating_backend(
    backend: str,
    *,
    hermes_home: str = "",
    agent_path: str = "",
    model: str = "",
    strict: bool = True,
) -> Any:
    """Construct a backend by name and wrap it so failures are loud.

    ``mock`` stays offline and unwrapped: it never calls out, and its scores are
    derived from recorded outcomes rather than from the skill, so it cannot show
    that an edit helped.
    """
    if backend in ("mock", ""):
        from hermes_skillopt.backend import HermesBackend
        return HermesBackend()

    if backend == "hermes":
        inner = HermesLlmBackend(model=model, hermes_home=hermes_home, agent_path=agent_path)
    else:
        from skillopt_sleep.backend import get_backend
        inner = get_backend(backend, model=model)

    return StrictBackend(inner) if strict else inner
