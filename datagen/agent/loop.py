"""The agent loop.

    observe -> plan (local LLM) -> act (tool) -> observe -> ... -> finish

Two planners:

  LLMPlanner        the local model picks the next tool from the objective, the
                    tool list, and a running scratchpad of observations. This is
                    the real agent.

  HeuristicPlanner  a deterministic state machine over the same tools. It runs
                    when no local model is reachable, and it is also the safety
                    net when the model returns garbage several times in a row.
                    Same tools, same outcome shape — just no reasoning.

Guard rails: a step budget, a consecutive-error budget, and a repeat detector
that blocks calling the same tool with the same arguments twice, which is the
failure mode small models fall into most often.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..config import Config
from ..llm import LocalLLM
from ..models import RunStats
from ..state import StateStore
from ..util import get_logger, now_iso, short_id, truncate
from . import tools as T

log = get_logger("agent")

# Consecutive steps that may produce nothing before the run is called stalled.
# Reconnaissance steps (search_web, assess_coverage) never gather by design, so
# this has to leave room for a couple of them in a row followed by the scrape
# that pays off.
STALL_LIMIT = 5

PLANNER_SYSTEM = """You are the controller of an autonomous dataset-building agent.
You gather technical material from the web, local documents, runbooks and
Confluence, then turn it into a training dataset using a LOCAL language model.

You decide ONE tool call at a time. Think about what the observations so far
tell you, then act. Rules:
- Gather material BEFORE calling build_dataset. Building with nothing gathered
  wastes a step.
- Prefer authoritative sources: vendor documentation, internal runbooks and
  Confluence over blog posts.
- Do not repeat a tool call that already succeeded, and do not retry one that
  failed for a structural reason (missing credentials, missing path).
- Call assess_coverage before deciding you are done.
- End with export_dataset, then finish.

Reply with JSON ONLY, in exactly this shape:
{"thought": "<one sentence of reasoning>", "tool": "<tool name>", "args": {...}}"""


@dataclass
class Step:
    n: int
    thought: str
    tool: str
    args: dict
    result: dict
    ok: bool

    def observation(self) -> str:
        """What the planner sees. Trimmed hard — a local model's context is
        precious and raw tool output is mostly noise."""
        payload = {k: v for k, v in self.result.items() if k not in ("manifest",)}
        return truncate(json.dumps(payload, ensure_ascii=False, default=str), 700)


@dataclass
class AgentRun:
    run_id: str
    steps: list[Step] = field(default_factory=list)
    finished: bool = False
    summary: str = ""

    def transcript(self, last: int = 8) -> str:
        lines = []
        for s in self.steps[-last:]:
            lines.append(
                f"[{s.n}] thought: {s.thought}\n"
                f"    action: {s.tool}({json.dumps(s.args, default=str)[:180]})\n"
                f"    result: {s.observation()}"
            )
        return "\n".join(lines) or "(no steps yet — this is the first decision)"


class HeuristicPlanner:
    """Fixed, sensible plan. Runs the same tools the LLM planner would."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._queue: list[tuple[str, dict]] = self._build_plan()

    def _build_plan(self) -> list[tuple[str, dict]]:
        plan: list[tuple[str, dict]] = []
        sources = self.cfg.sources

        if sources.get("runbooks"):
            plan.append(("ingest_runbooks", {}))
        if sources.get("files"):
            plan.append(("ingest_files", {}))

        conf = sources.get("confluence") or []
        conf_blocks = conf if isinstance(conf, list) else [conf]
        for block in conf_blocks:
            if block.get("enabled", True):
                plan.append(("search_confluence", {"spaces": block.get("spaces", []),
                                                   "limit": block.get("max_pages", 25)}))

        web = sources.get("web") or []
        web_blocks = web if isinstance(web, list) else [web]
        for block in web_blocks:
            if not block.get("enabled", True):
                continue
            for seed in block.get("seeds", []):
                plan.append(("crawl_site", {"url": seed,
                                            "max_pages": block.get("max_pages", 15),
                                            "max_depth": block.get("max_depth", 2)}))

        kw = sources.get("keywords") or {}
        if kw.get("enabled", True):
            for term in kw.get("terms", []):
                plan.append(("scrape_keyword", {"keyword": term,
                                                "max_pages": kw.get("max_pages_per_keyword", 6)}))

        plan += [
            ("build_dataset", {}),
            ("assess_coverage", {}),
            ("export_dataset", {}),
            ("finish", {"summary": "deterministic plan completed"}),
        ]
        return plan

    def next_action(self, run: AgentRun, ctx: T.ToolContext) -> tuple[str, str, dict]:
        if not self._queue:
            return "nothing left in the plan", "finish", {"summary": "plan exhausted"}
        tool, args = self._queue.pop(0)
        return f"plan step: {tool}", tool, args


class LLMPlanner:
    """Local model chooses the next tool."""

    def __init__(self, cfg: Config, llm: LocalLLM) -> None:
        self.cfg = cfg
        self.llm = llm
        self.consecutive_bad = 0

    def next_action(self, run: AgentRun, ctx: T.ToolContext) -> tuple[str, str, dict] | None:
        prompt = f"""OBJECTIVE:
{self.cfg.agent.objective.strip()}

AVAILABLE TOOLS:
{T.schema_text()}

CURRENT STATE:
- step {len(run.steps) + 1} of at most {self.cfg.agent.max_steps}
- documents gathered and not yet built: {len(ctx.documents)}
- chunks produced this run: {len(ctx.chunks)}
- records produced this run: {len(ctx.records)}
- leads saved for next run: {len(ctx.state.pending())}

RECENT STEPS:
{run.transcript()}

What is the single next tool call? JSON only."""

        data = self.llm.complete_json(prompt, system=PLANNER_SYSTEM, temperature=0.3)
        if not isinstance(data, dict) or not data.get("tool"):
            self.consecutive_bad += 1
            log.warning("planner returned no usable action (%d in a row)", self.consecutive_bad)
            return None

        tool = str(data["tool"]).strip()
        args = data.get("args") or data.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}
        self.consecutive_bad = 0
        return str(data.get("thought", ""))[:300], tool, args


class Agent:
    def __init__(self, cfg: Config, state: StateStore, llm: LocalLLM) -> None:
        self.cfg = cfg
        self.state = state
        self.llm = llm

    def run(self, objective: str | None = None, max_steps: int | None = None) -> AgentRun:
        cfg = self.cfg
        if objective:
            cfg.agent.objective = objective
        budget = max_steps or cfg.agent.max_steps

        stats = RunStats(run_id=short_id(now_iso(), "agent"))
        run = AgentRun(run_id=stats.run_id)
        cfg.ensure_dirs()
        self.state.start_run(run.run_id, "agent")

        ctx = T.ToolContext(
            cfg=cfg, state=self.state, llm=self.llm, stats=stats,
            documents=[], chunks=[], records=[],
        )

        use_llm = self.llm.available()
        planner: Any = LLMPlanner(cfg, self.llm) if use_llm else HeuristicPlanner(cfg)
        fallback = HeuristicPlanner(cfg)

        log.info("=== agent %s | planner: %s ===",
                 run.run_id, "local model " + cfg.llm.model if use_llm else "heuristic (no LLM)")
        log.info("objective: %s", truncate(cfg.agent.objective.strip(), 200))

        errors = 0
        attempted: set[str] = set()
        last_progress: tuple[int, int, int] | None = None
        stalled = 0

        for n in range(1, budget + 1):
            action = planner.next_action(run, ctx)

            # The model failed to produce an action — fall back for this step.
            if action is None:
                if getattr(planner, "consecutive_bad", 0) >= 3:
                    log.warning("planner unreliable — switching to the heuristic plan")
                    planner = fallback
                action = fallback.next_action(run, ctx)

            thought, tool, args = action

            # Repeat detector: same tool + same args is always a wasted step.
            # NB it must not `continue` past the stall check below — a planner
            # that proposes one identical call over and over is the clearest
            # stall there is, and skipping the check let exactly that burn a
            # whole 24-step budget on 22 skipped search_web calls.
            signature = f"{tool}:{json.dumps(args, sort_keys=True, default=str)}"
            repeated = signature in attempted and tool not in ("assess_coverage", "finish")

            if repeated:
                log.info("[%d] skipping repeat call %s", n, tool)
                run.steps.append(
                    Step(n, thought, tool, args,
                         {"ok": False, "error": "this exact call was already made; choose a different action"},
                         False)
                )
            else:
                attempted.add(signature)

                log.info("[%d/%d] %s → %s", n, budget, truncate(thought, 90) or "(no thought)", tool)
                result = T.call(ctx, tool, args)
                ok = bool(result.get("ok"))
                run.steps.append(Step(n, thought, tool, args, result, ok))

                if ok:
                    errors = 0
                    log.info("      ✓ %s", truncate(json.dumps(
                        {k: v for k, v in result.items() if k not in ("ok", "manifest")},
                        default=str), 160))
                else:
                    errors += 1
                    log.warning("      ✗ %s", result.get("error", "failed"))
                    if errors >= cfg.agent.max_tool_errors:
                        log.error("too many consecutive tool failures — stopping")
                        break

                if result.get("done"):
                    run.finished = True
                    run.summary = result.get("summary", "")
                    break

            # Stall detector. The repeat detector only catches an *identical*
            # call, and assess_coverage is exempt from it because coverage
            # genuinely changes as material arrives — so a planner that keeps
            # reassessing an unchanged dataset sails past both. Observed
            # spending the last third of a budget that way.
            #
            # The limit is not tighter because searching and assessing never
            # gather anything by definition: `search → assess → search → scrape`
            # is a normal, productive rhythm, and a limit of 3 cut a real run off
            # at the search that had just surfaced a new URL. Only the scrape
            # would have shown whether it was worth anything.
            progress = (len(ctx.documents), len(ctx.chunks), len(ctx.records))
            if progress == last_progress:
                stalled += 1
                if stalled >= STALL_LIMIT:
                    log.info(
                        "no new documents, chunks or records in %d steps — "
                        "finishing instead of burning the remaining budget", stalled
                    )
                    run.summary = "stopped early: the plan stopped making progress"
                    break
            else:
                stalled = 0
                last_progress = progress
        else:
            log.info("step budget exhausted")

        # Always land the work, whatever the planner decided.
        if ctx.documents:
            log.info("finalising %d ungathered documents", len(ctx.documents))
            T.call(ctx, "build_dataset", {})
        if ctx.records or ctx.chunks:
            T.call(ctx, "export_dataset", {})

        if cfg.agent.propose_new_sources and use_llm:
            self._propose_next(ctx, run)

        stats.finished_at = now_iso()
        self.state.finish_run(run.run_id, {
            "steps": len(run.steps),
            "records": stats.records,
            "documents": stats.docs_fetched,
            "summary": run.summary,
        }, ok=True)

        log.info("=== agent finished after %d steps: %s ===",
                 len(run.steps), run.summary or stats.summary())
        return run

    def _propose_next(self, ctx: T.ToolContext, run: AgentRun) -> None:
        """Ask the local model what is missing, and store it for the next run.
        This is the self-improvement loop."""
        coverage = T.call(ctx, "assess_coverage", {})
        data = self.llm.complete_json(
            f"""You just built part of a dataset with this objective:
{self.cfg.agent.objective.strip()}

Coverage report:
{json.dumps(coverage, indent=2, default=str)[:2000]}

What is MISSING? Propose specific search keywords and documentation URLs that
would close the biggest gaps. Be concrete — "MLIS endpoint 503 troubleshooting"
not "more troubleshooting". Do not repeat topics already well covered.

JSON only:
{{"keywords": ["..."], "urls": ["..."], "reason": "<one sentence>"}}""",
            system="You plan dataset coverage. Reply with JSON only.",
            temperature=0.5,
        )
        if isinstance(data, dict):
            added = T.call(ctx, "propose_sources", data)
            n = len(added.get("added", {}).get("keywords", [])) + len(added.get("added", {}).get("urls", []))
            if n:
                log.info("queued %d new leads for the next run", n)


def run_agent(cfg: Config, objective: str | None = None, max_steps: int | None = None) -> AgentRun:
    llm = LocalLLM(cfg.llm)
    with StateStore(cfg.state_db) as state:
        return Agent(cfg, state, llm).run(objective, max_steps)
