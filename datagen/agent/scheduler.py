"""Self-updating scheduler.

`datagen watch` keeps the dataset current without anyone driving it:

  * every `interval_min`, re-run the agent (or the plain pipeline)
  * documents whose content hash is unchanged are skipped, so a routine cycle
    is cheap and mostly no-ops
  * changed pages are re-chunked and their stale records dropped
  * leads the agent proposed last cycle are picked up this cycle, so coverage
    widens on its own
  * every `full_rebuild_every` cycles, ignore caches and rebuild from scratch,
    which repairs any drift accumulated by incremental runs

It is a plain foreground loop — run it under NSSM/Task Scheduler on Windows or
systemd/cron elsewhere. Interrupting it is always safe: state lives in SQLite
and is committed after every step.
"""

from __future__ import annotations

import random
import signal
import time
from dataclasses import dataclass
from typing import Any

from ..config import Config
from ..llm import LocalLLM
from ..pipeline import Pipeline
from ..state import StateStore
from ..util import get_logger, now_iso

log = get_logger("scheduler")


@dataclass
class Scheduler:
    cfg: Config
    use_agent: bool = True
    once: bool = False

    _stop: bool = False

    def _install_signals(self) -> None:
        def handler(signum: int, frame: Any) -> None:
            log.info("signal %s received — finishing this cycle then stopping", signum)
            self._stop = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass  # not on the main thread, or unsupported on this platform

    def _sleep(self, seconds: float) -> None:
        """Interruptible sleep so Ctrl-C does not wait out the whole interval."""
        deadline = time.time() + seconds
        while time.time() < deadline and not self._stop:
            time.sleep(min(2.0, deadline - time.time()))

    def run(self) -> None:
        self._install_signals()
        cfg = self.cfg
        interval = max(60, cfg.schedule.interval_min * 60)

        with StateStore(cfg.state_db) as state:
            cycle = int(state.get("scheduler.cycle", 0))
            log.info(
                "watch started: every %d min, full rebuild every %d cycles, mode=%s",
                cfg.schedule.interval_min, cfg.schedule.full_rebuild_every,
                "agent" if self.use_agent else "pipeline",
            )

            while not self._stop:
                cycle += 1
                full = (
                    cfg.schedule.full_rebuild_every > 0
                    and cycle % cfg.schedule.full_rebuild_every == 0
                )
                log.info("── cycle %d (%s) at %s ──",
                         cycle, "FULL rebuild" if full else "incremental", now_iso())

                started = time.time()
                try:
                    self._cycle(state, full)
                except KeyboardInterrupt:
                    log.info("interrupted")
                    break
                except Exception as e:
                    log.error("cycle %d failed: %s", cycle, e, exc_info=True)
                    state.set("scheduler.last_error", f"{now_iso()}: {e}")

                state.set("scheduler.cycle", cycle)
                state.set("scheduler.last_run", now_iso())
                log.info("cycle %d took %.1fs", cycle, time.time() - started)

                if self.once or self._stop:
                    break

                jitter = interval * (cfg.schedule.jitter_pct / 100.0)
                delay = interval + random.uniform(-jitter, jitter)
                log.info("next cycle in %.0f min", delay / 60)
                self._sleep(delay)

        log.info("watch stopped after %d cycles", cycle)

    def _cycle(self, state: StateStore, full: bool) -> None:
        llm = LocalLLM(self.cfg.llm)
        if not llm.available(recheck=True):
            log.warning(
                "local model unreachable this cycle — running extractive only. "
                "Start it with `ollama serve` and the next cycle will use it."
            )

        if self.use_agent and self.cfg.agent.enabled and llm.available():
            # The agent picks up proposed leads and widens coverage on its own.
            from .loop import Agent

            Agent(self.cfg, state, llm).run()
        else:
            Pipeline(self.cfg, state, llm).run(full=full, export=True)


def watch(cfg: Config, use_agent: bool = True, once: bool = False) -> None:
    Scheduler(cfg, use_agent=use_agent, once=once).run()
