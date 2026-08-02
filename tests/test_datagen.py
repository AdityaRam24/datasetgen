"""Unit tests for the parts that have to be right.

Run:  python -m pytest tests -q          (or: python tests/test_datagen.py)

Nothing here touches the network or the local LLM — those paths are covered by
`python -m datagen doctor` instead.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datagen.analyze import (                                     # noqa: E402
    LengthStats, analyze, check_leakage, dataset_card, find_degenerate, percentile,
)
from datagen.chunking import chunk_document                      # noqa: E402
from datagen.config import ChunkingConfig, LLMConfig, QualityConfig, load_config  # noqa: E402
from datagen.connectors.files import read_file                    # noqa: E402
from datagen.connectors import parsers as parsers_mod                # noqa: E402
from datagen.connectors.parsers import parse_html, _postprocess_pdf  # noqa: E402
from datagen.connectors.runbooks import normalize, parse_runbook  # noqa: E402
from datagen.dedupe import Deduper, hamming, simhash             # noqa: E402
from datagen.exporters import (                                   # noqa: E402
    build_glossary, glossary_markdown, split_by_document, to_alpaca, to_chatml,
)
from datagen.connectors.search import (                           # noqa: E402
    SearchResult, rank_results, searxng_available,
)
from datagen.generators import (                                  # noqa: E402
    _extractive_glossary, is_useful_term, system_prompt,
)
from datagen.learn import case_record, document_from_text, pair_record  # noqa: E402
from datagen.models import Chunk, Document, Record               # noqa: E402
from datagen.quality import grounding_score, heuristic_check     # noqa: E402
from datagen.state import StateStore, _to_signed64, _to_unsigned64  # noqa: E402
from datagen.web import (                                         # noqa: E402
    mask_secret, safe_upload_path, update_env, update_keywords, update_project, update_toml,
)
from datagen.util import (                                        # noqa: E402
    clean_text, estimate_tokens, extract_json, tokenize,
)


def make_doc(text: str, title: str = "Doc", kind: str = "markdown") -> Document:
    return Document.make(title=title, url=f"file:///{title}", text=text, kind=kind, source="test")


class TestUtil(unittest.TestCase):
    def test_clean_text_collapses_whitespace_but_keeps_paragraphs(self):
        out = clean_text("a  \t b\r\n\n\n\nc")
        self.assertEqual(out, "a b\n\nc")

    def test_tokenize_drops_stopwords_and_short_tokens(self):
        # Paths and flags survive intact — they are what grounding checks on.
        self.assertEqual(tokenize("The MLIS endpoint is at /v1/models"),
                         ["mlis", "endpoint", "/v1/models"])
        # `=` splits the token and single characters are dropped, so the flag
        # name survives but its `0` value does not.
        self.assertEqual(tokenize("Set --replicas=0 on deploy"),
                         ["set", "--replicas", "deploy"])

    def test_extract_json_handles_fenced_and_prefixed_output(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json('Sure! Here you go: {"a": [1,2]} hope that helps'),
                         {"a": [1, 2]})
        self.assertEqual(extract_json('[{"x": "}"}]'), [{"x": "}"}])
        self.assertIsNone(extract_json("no json at all"))


class TestChunking(unittest.TestCase):
    cfg = ChunkingConfig(max_chars=300, overlap=40, min_chars=50)

    def test_headings_become_chunk_context(self):
        doc = make_doc(
            "# Install\n\n" + "Steps for installing the platform. " * 8 +
            "\n\n## Upgrade\n\n" + "Steps for upgrading the platform. " * 8
        )
        chunks = chunk_document(doc, self.cfg)
        self.assertTrue(chunks)
        headings = {c.heading for c in chunks}
        self.assertIn("Install", headings)
        self.assertTrue(any("Upgrade" in h for h in headings))
        # Each chunk carries its heading so it is self-describing to the LLM.
        for c in chunks:
            if c.heading:
                self.assertIn(c.heading.split(" › ")[-1], c.text)

    def test_oversized_paragraph_is_split(self):
        doc = make_doc("word " * 400)
        chunks = chunk_document(doc, self.cfg)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c.text), self.cfg.max_chars + 80)

    def test_empty_document_yields_nothing(self):
        self.assertEqual(chunk_document(make_doc("   \n\n  "), self.cfg), [])

    def test_short_single_section_is_kept(self):
        doc = make_doc("A short but real note about GPU scheduling.")
        self.assertEqual(len(chunk_document(doc, self.cfg)), 1)


class TestDedupe(unittest.TestCase):
    def test_simhash_is_close_for_near_identical_text(self):
        a = simhash("The MLIS endpoint returned 503 Service Unavailable on node 4")
        b = simhash("The MLIS endpoint returned 503 Service Unavailable on node 7")
        c = simhash("Keycloak group mappings are applied at namespace granularity")
        self.assertLess(hamming(a, b), hamming(a, c))

    def test_exact_duplicate_is_caught(self):
        d = Deduper(use_near=False, use_semantic=False)
        self.assertIsNone(d.check("1", "identical text here"))
        self.assertEqual(d.check("2", "identical  TEXT here"), "exact")
        self.assertEqual(d.stats.exact, 1)

    def test_near_duplicate_is_caught(self):
        d = Deduper(max_distance=8, use_semantic=False)
        base = "Scale the endpoint to zero replicas to free the GPU allocation. " * 4
        self.assertIsNone(d.check("1", base))
        self.assertIsNotNone(d.check("2", base + "Extra trailing sentence."))

    def test_distinct_text_is_kept(self):
        d = Deduper(use_semantic=False)
        self.assertIsNone(d.check("1", "Keycloak brokers authentication for all AI services."))
        self.assertIsNone(d.check("2", "The lakehouse presents an S3-compatible object interface."))
        self.assertEqual(d.stats.total, 0)

    def test_duplicate_questions_are_dropped(self):
        chunk = Chunk.make(make_doc("body text"), "body text", 0)
        recs = [
            Record.make("qa", "How do I fix a 503 from MLIS?", "answer one", chunk),
            Record.make("qa", "How do I fix a 503 from MLIS?", "answer two", chunk),
            Record.make("qa", "How do I rotate lakehouse credentials?", "answer three", chunk),
        ]
        self.assertEqual(len(Deduper().dedupe_records(recs)), 2)


class TestRunbooks(unittest.TestCase):
    RUNBOOK = """# Endpoint down

## Symptom
Requests return 503.

## Steps
1. Check the pods.
2. Read the init container logs.
3. Refresh the bucket secret.

```
kubectl get pods -n mlis
```

## Rollback
Run `kubectl rollout undo`.
"""

    def test_sections_and_steps_are_extracted(self):
        parsed = parse_runbook(self.RUNBOOK, "Endpoint down")
        self.assertIn("503", parsed["symptom"])
        self.assertEqual(len(parsed["steps"]), 3)
        self.assertEqual(parsed["steps"][0], "Check the pods.")
        self.assertIn("kubectl get pods -n mlis", parsed["commands"])
        self.assertIn("rollout undo", parsed["rollback"])

    def test_normalized_output_preserves_step_order(self):
        out = normalize(parse_runbook(self.RUNBOOK, "Endpoint down"), "Endpoint down")
        self.assertLess(out.index("1. Check the pods."), out.index("2. Read"))
        self.assertIn("## Rollback", out)


class TestParsers(unittest.TestCase):
    def test_html_extraction_drops_script_and_nav(self):
        html = """<html><head><title>Docs</title><style>b{}</style></head>
        <body><nav>Menu</nav><script>evil()</script>
        <main><h1>GPU sizing</h1><p>Plan for one node of headroom.</p></main></body></html>"""
        title, text = parse_html(html)
        self.assertIn("GPU sizing", text)
        self.assertIn("headroom", text)
        self.assertNotIn("evil()", text)
        if title:
            self.assertIn("Docs", title)

    def test_pdf_postprocess_rejoins_hyphenated_and_wrapped_lines(self):
        out = _postprocess_pdf("The end-\npoint was un-\navailable\nbecause of quota.\n12\n")
        self.assertIn("endpoint", out)
        self.assertIn("unavailable because", out)
        self.assertNotIn("\n12", out)


class TestImages(unittest.TestCase):
    """Image support without a vision model — the describer is injected, so the
    whole path is testable with no LLM and no network."""

    PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

    def tearDown(self):
        parsers_mod.set_image_describer(None)

    def test_images_are_recognised_as_a_kind(self):
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            self.assertEqual(parsers_mod.EXTENSION_MAP.get(ext), "image", ext)

    def test_no_vision_model_yields_no_text_rather_than_an_error(self):
        parsers_mod.set_image_describer(None)
        title, text = parsers_mod.parse_bytes(self.PNG, "image", url="shot.png")
        self.assertEqual(text, "")
        self.assertEqual(title, "")

    def test_description_becomes_the_document_text(self):
        seen = {}

        def fake(data, url, mime):
            seen.update(bytes=len(data), url=url, mime=mime)
            return "The dialog reads: MLIS endpoint returned 503 Service Unavailable."

        parsers_mod.set_image_describer(fake)
        _, text = parsers_mod.parse_bytes(self.PNG, "image", url="http://x/mlis-503.png")
        self.assertIn("503 Service Unavailable", text)
        self.assertEqual(seen["mime"], "image/png")
        self.assertEqual(seen["bytes"], len(self.PNG))

    def test_mime_follows_the_extension(self):
        got = {}
        parsers_mod.set_image_describer(lambda d, u, m: got.setdefault("mime", m) or "ok text")
        parsers_mod.parse_bytes(self.PNG, "image", url="/a/b/diagram.jpg?v=2")
        self.assertEqual(got["mime"], "image/jpeg")

    def test_a_failing_vision_model_does_not_break_the_build(self):
        def boom(data, url, mime):
            raise RuntimeError("model exploded")

        parsers_mod.set_image_describer(boom)
        _, text = parsers_mod.parse_bytes(self.PNG, "image", url="x.png")
        self.assertEqual(text, "")

    def test_short_description_survives_chunking(self):
        # A screenshot description is often well under chunking.min_chars; it is
        # the document's entire content, so it must not be dropped.
        doc = Document.make(
            title="mlis-503.png",
            url="file:///corpus/docs/mlis-503.png",
            text="Image: mlis-503.png\n\nThe console shows 503 Service Unavailable.",
            kind="image",
            source="local-docs",
        )
        chunks = chunk_document(doc, ChunkingConfig())
        self.assertEqual(len(chunks), 1)
        self.assertIn("503", chunks[0].text)


class TestQuality(unittest.TestCase):
    cfg = QualityConfig()

    def _rec(self, q: str, a: str, context: str = "") -> Record:
        chunk = Chunk.make(make_doc(context or a), context or a, 0)
        return Record.make("qa", q, a, chunk)

    def test_meta_questions_are_rejected(self):
        v = heuristic_check(self._rec("What does the source text say about GPUs?", "x" * 100), self.cfg)
        self.assertFalse(v.ok)

    def test_non_answers_are_rejected(self):
        v = heuristic_check(
            self._rec("How do I free GPU capacity?",
                      "That is not specified in the provided document at all, sorry."),
            self.cfg,
        )
        self.assertFalse(v.ok)

    def test_good_pair_passes(self):
        v = heuristic_check(
            self._rec(
                "How do I free GPU capacity on a PCAI cluster?",
                "Scale an idle endpoint to zero replicas with "
                "`kubectl -n mlis scale deploy <endpoint> --replicas=0`, which releases "
                "its GPU allocation for other workloads.",
            ),
            self.cfg,
        )
        self.assertTrue(v.ok, v.reason)

    def test_unbalanced_code_fence_is_rejected(self):
        v = heuristic_check(self._rec("How do I check pods?", "Run this:\n```\nkubectl get pods\n" ), self.cfg)
        self.assertFalse(v.ok)

    def test_grounding_score_separates_grounded_from_invented(self):
        context = ("Scale the endpoint to zero replicas with kubectl scale deploy "
                   "--replicas=0 to release the GPU allocation.")
        grounded = self._rec("How do I release a GPU?",
                             "Use kubectl scale deploy --replicas=0 to release the GPU allocation.",
                             context)
        invented = self._rec("How do I release a GPU?",
                             "Edit /etc/nvidia/quota.conf and restart the licence daemon on port 7070.",
                             context)
        self.assertGreater(grounding_score(grounded), grounding_score(invented))
        self.assertGreater(grounding_score(grounded), 0.5)


class TestState(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.state = StateStore(Path(self.dir.name) / "state.db")

    def tearDown(self):
        self.state.close()
        self.dir.cleanup()

    def test_signed_roundtrip_covers_the_full_unsigned_range(self):
        for value in (0, 1, (1 << 63) - 1, 1 << 63, (1 << 64) - 1):
            self.assertEqual(_to_unsigned64(_to_signed64(value)), value)

    def test_simhash_survives_the_database(self):
        h = simhash("a" * 500)          # reliably sets the high bit sometimes
        self.state.add_chunk("c1", "d1", "hash", h)
        stored = dict(self.state.all_simhashes())
        self.assertEqual(stored["c1"], h)

    def test_document_change_detection(self):
        doc = make_doc("original body text")
        self.assertTrue(self.state.upsert_document(doc))     # new
        self.assertFalse(self.state.upsert_document(doc))    # unchanged
        changed = make_doc("edited body text")
        changed.id = doc.id                                  # same URL, new content
        changed.content_hash = "different"
        self.assertTrue(self.state.upsert_document(changed))

    def test_doc_has_chunks_gates_the_skip(self):
        doc = make_doc("body")
        self.state.upsert_document(doc)
        self.assertFalse(self.state.doc_has_chunks(doc.id))
        self.state.add_chunk("c1", doc.id, "h", 1)
        self.assertTrue(self.state.doc_has_chunks(doc.id))

    def test_proposals_are_deduplicated(self):
        self.assertTrue(self.state.propose("mlis 503", "keyword", "gap"))
        self.assertFalse(self.state.propose("mlis 503", "keyword", "gap again"))
        self.assertEqual(len(self.state.pending("keyword")), 1)
        self.state.mark("mlis 503", "done")
        self.assertEqual(len(self.state.pending("keyword")), 0)


class TestExporters(unittest.TestCase):
    def _records(self, n: int, docs: int) -> list[Record]:
        out = []
        for i in range(n):
            doc = make_doc(f"body {i}", title=f"doc{i % docs}")
            chunk = Chunk.make(doc, f"body {i}", 0)
            out.append(Record.make("qa", f"question {i}?", f"answer {i}", chunk))
        return out

    def test_split_never_puts_one_document_on_both_sides(self):
        train, evalset = split_by_document(self._records(60, 10), 0.8, seed=1)
        self.assertTrue(train and evalset)
        self.assertFalse({r.source_url for r in train} & {r.source_url for r in evalset})
        self.assertEqual(len(train) + len(evalset), 60)
        self.assertGreaterEqual(len(train), len(evalset))

    def test_split_stays_train_heavy_with_few_documents(self):
        # Regression: two lopsided documents used to invert the split entirely,
        # putting 15 of 18 records into eval.
        records = self._records(3, 1)
        for i in range(15):
            doc = make_doc(f"other {i}", title="doc-big")
            records.append(Record.make("qa", f"big q {i}?", f"big a {i}", Chunk.make(doc, "t", 0)))
        train, evalset = split_by_document(records, 0.9, seed=42)
        self.assertEqual(len(train) + len(evalset), 18)
        self.assertGreater(len(train), len(evalset))
        self.assertTrue(evalset)
        self.assertFalse({r.source_url for r in train} & {r.source_url for r in evalset})

    def test_split_with_a_single_document_keeps_everything_in_train(self):
        train, evalset = split_by_document(self._records(10, 1), 0.9, seed=1)
        self.assertEqual(len(train), 10)
        self.assertEqual(evalset, [])

    def test_alpaca_shape_and_citation(self):
        rec = self._records(1, 1)[0]
        row = to_alpaca(rec, cite=True)
        self.assertEqual(set(row), {"instruction", "input", "output"})
        self.assertIn("Source:", row["output"])
        self.assertNotIn("Source:", to_alpaca(rec, cite=False)["output"])

    def test_chatml_shape(self):
        row = to_chatml(self._records(1, 1)[0], system="sys")
        roles = [m["role"] for m in row["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant"])


class TestGlossary(unittest.TestCase):
    def _entry(self, term, definition, score=1.0, url="file:///a", aliases=None):
        chunk = Chunk.make(make_doc(definition, title="Src"), definition, 0)
        chunk.url = url
        rec = Record.make("glossary", f"What is {term}?", definition, chunk,
                          meta={"term": term, "aliases": aliases or []})
        rec.score = score
        return rec

    def test_system_prompt_follows_the_configured_subject(self):
        cfg = load_config()
        cfg.description = "PostgreSQL administration and performance tuning"
        prompt = system_prompt(cfg)
        self.assertIn("PostgreSQL administration", prompt)
        self.assertIn("never invent facts", prompt)
        # No config: the persona still works, just without a subject line.
        self.assertNotIn("assistant about:", system_prompt())

    def test_useful_term_filter(self):
        for good in ("MLIS", "CrashLoopBackOff", "control plane", "nvidia.com/gpu"):
            self.assertTrue(is_useful_term(good), good)
        for bad in ("", "a", "the endpoint", "This is a very long sentence masquerading as a term"):
            self.assertFalse(is_useful_term(bad), bad)

    def test_terms_merge_across_chunks_keeping_the_best_definition(self):
        entries = build_glossary([
            self._entry("MLIS", "MLIS is a short one.", score=0.6, url="file:///a"),
            self._entry("mlis", "MLIS is a service that serves inference endpoints on the "
                                "GPU worker pool.", score=0.95, url="file:///b"),
            self._entry("MLIS", "MLIS is another mention.", score=0.5, url="file:///c"),
        ])
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertIn("serves inference endpoints", entry["definition"])
        self.assertEqual(entry["mentions"], 3)
        self.assertEqual(len(entry["sources"]), 3)   # every source is cited

    def test_aliases_are_collected_and_self_alias_dropped(self):
        entries = build_glossary([
            self._entry("MLDM", "MLDM stands for Machine Learning Data Management.",
                        aliases=["Machine Learning Data Management", "MLDM"]),
        ])
        self.assertEqual(entries[0]["aliases"], ["Machine Learning Data Management"])

    def test_entries_are_sorted_alphabetically(self):
        entries = build_glossary([
            self._entry("Zookeeper", "Zookeeper is a coordination service used here."),
            self._entry("Argo", "Argo is a workflow engine used by the platform."),
        ])
        self.assertEqual([e["term"] for e in entries], ["Argo", "Zookeeper"])

    def test_markdown_has_headings_sources_and_az_groups(self):
        cfg = load_config()
        md = glossary_markdown(build_glossary([
            self._entry("Argo", "Argo is a workflow engine used by the platform."),
            self._entry("Zookeeper", "Zookeeper is a coordination service used here."),
        ]), cfg)
        self.assertIn("### Argo", md)
        self.assertIn("## A", md)
        self.assertIn("## Z", md)
        self.assertIn("Sources:", md)
        self.assertLess(md.index("### Argo"), md.index("### Zookeeper"))

    def test_extractive_glossary_finds_acronyms_and_copulas(self):
        body = (
            "MLIS (Machine Learning Inference Service) hosts endpoints.\n"
            "CrashLoopBackOff is a pod state that means the container keeps restarting "
            "after repeated failures.\n"
            "The weather was nice that day.\n"
        )
        chunk = Chunk.make(make_doc(body), body, 0)
        terms = {(r.meta or {}).get("term") for r in _extractive_glossary(chunk, body)}
        self.assertIn("MLIS", terms)
        self.assertIn("CrashLoopBackOff", terms)

    def test_extractive_rejects_a_parenthetical_that_is_not_an_expansion(self):
        body = "The GPU (which we installed last year) is idle right now."
        chunk = Chunk.make(make_doc(body), body, 0)
        terms = {(r.meta or {}).get("term") for r in _extractive_glossary(chunk, body)}
        self.assertNotIn("GPU", terms)


class TestLearn(unittest.TestCase):
    def test_text_becomes_a_document_with_input_provenance(self):
        doc = document_from_text(
            "MLIS endpoints hold their GPU allocation until scaled to zero replicas, "
            "which is the usual cause of capacity errors.",
            tags=["gpu"],
        )
        self.assertIsNotNone(doc)
        self.assertTrue(doc.url.startswith("input://"))
        self.assertIn("user-input", doc.tags)
        self.assertIn("gpu", doc.tags)

    def test_title_is_derived_from_the_first_line(self):
        doc = document_from_text("# GPU scheduling notes\n\n" + "Body text here. " * 8)
        self.assertEqual(doc.title, "GPU scheduling notes")

    def test_trivially_short_input_is_refused(self):
        self.assertIsNone(document_from_text("too short"))

    def test_human_pair_is_trusted_and_traceable(self):
        rec = pair_record("How do I free a GPU?", "Scale the endpoint to zero replicas.",
                          tags=["ops"])
        self.assertEqual(rec.score, 1.0)
        self.assertEqual(rec.generator, "human:input")
        self.assertTrue(rec.source_url.startswith("human://"))
        self.assertIn("human-authored", rec.tags)
        self.assertIn("ops", rec.tags)
        self.assertFalse(rec.quarantined)

    def test_human_pair_survives_the_quality_gate_untouched(self):
        # The gate must never quarantine a human correction for being
        # "ungrounded" — there is no source chunk to be grounded against.
        rec = pair_record(
            "How do I free a GPU allocation?",
            "Scale the idle endpoint to zero replicas; the GPU is released immediately.",
        )
        verdict = heuristic_check(rec, QualityConfig())
        self.assertTrue(verdict.ok, verdict.reason)

    def test_solved_case_becomes_a_troubleshooting_record(self):
        rec = case_record("endpoint 503 after upgrade", "bucket credentials expired; recreate the secret")
        self.assertEqual(rec.kind, "troubleshooting")
        self.assertIn("503 after upgrade", rec.instruction)
        self.assertIn("solved-case", rec.tags)

    def test_relative_file_path_is_accepted(self):
        # Regression: read_file called Path.as_uri(), which raises on a
        # relative path, so `learn --file ./x.md` and `ingest ./docs` failed.
        import os

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "note.md"
            path.write_text("# Notes\n\n" + "Real content about GPU scheduling. " * 6,
                            encoding="utf-8")
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                doc = read_file(Path("note.md"), "test")
            finally:
                os.chdir(cwd)

        self.assertIsNotNone(doc)
        self.assertTrue(doc.url.startswith("file:///"))

    def test_identical_input_produces_a_stable_id(self):
        a = pair_record("How do I free a GPU?", "Scale to zero.")
        b = pair_record("How do I free a GPU?", "Scale to zero.")
        self.assertEqual(a.id, b.id)   # re-learning the same thing is idempotent


class TestSearch(unittest.TestCase):
    def test_ranking_prefers_primary_docs_and_drops_junk(self):
        results = [
            SearchResult("https://pinterest.com/pin/1", "MLIS pin", "", 0),
            SearchResult("https://randomblog.dev/post", "Some MLIS post", "mlis endpoint", 1),
            SearchResult("https://docs.example.com/mlis/endpoints",
                         "MLIS endpoints", "mlis endpoint troubleshooting", 2),
        ]
        ranked = rank_results(results, "mlis endpoint troubleshooting")
        self.assertEqual(len(ranked), 2)                       # pinterest dropped
        self.assertIn("docs.example.com", ranked[0].url)       # primary docs first

    def test_ranking_penalises_archive_pages(self):
        ranked = rank_results([
            SearchResult("https://docs.example.com/guide/mlis", "MLIS guide", "mlis", 0),
            SearchResult("https://docs.example.com/tag/12", "Tag page", "mlis", 0),
        ], "mlis")
        self.assertIn("/guide/", ranked[0].url)

    def test_searxng_unreachable_is_reported_not_raised(self):
        # A stopped container must degrade, never abort a run.
        self.assertFalse(searxng_available("http://127.0.0.1:9"))
        self.assertFalse(searxng_available(""))


class TestAnalyze(unittest.TestCase):
    def _rec(self, q: str, a: str, kind: str = "qa", url: str = "file:///a") -> Record:
        doc = make_doc(a, title="Src")
        doc.url = url
        chunk = Chunk.make(doc, a, 0)
        chunk.url = url
        rec = Record.make(kind, q, a, chunk)
        rec.score = 1.0
        return rec

    def test_percentile_interpolates(self):
        vals = [1, 2, 3, 4, 5]
        self.assertEqual(percentile(vals, 0.0), 1)
        self.assertEqual(percentile(vals, 0.5), 3)
        self.assertEqual(percentile(vals, 1.0), 5)
        self.assertEqual(percentile([], 0.5), 0.0)
        self.assertEqual(percentile([7], 0.9), 7)

    def test_length_stats_ordering(self):
        stats = LengthStats.of([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        self.assertEqual(stats.count, 10)
        self.assertEqual(stats.minimum, 1)
        self.assertEqual(stats.maximum, 100)
        self.assertLessEqual(stats.p50, stats.p90)
        self.assertLessEqual(stats.p90, stats.p95)

    def test_token_estimate_is_conservative_for_dense_text(self):
        # Code/CLI text tokenizes denser than prose; the word rule must win so
        # truncation warnings do not under-count.
        dense = "kubectl -n mlis scale deploy x --replicas=0 && kubectl get po -o wide"
        self.assertGreaterEqual(estimate_tokens(dense), len(dense) // 4)
        self.assertEqual(estimate_tokens(""), 0)

    def test_truncation_is_detected(self):
        long_answer = "word " * 3000
        cfg = load_config()
        rep = analyze([self._rec("A long one?", long_answer)], cfg, max_seq_len=512)
        self.assertEqual(rep.over_limit, 1)
        self.assertEqual(rep.over_limit_pct, 100.0)
        self.assertTrue(any("TRUNCATED" in w for w in rep.warnings))
        self.assertTrue(rep.longest_examples)

    def test_no_truncation_warning_when_everything_fits(self):
        cfg = load_config()
        rep = analyze([self._rec("Short?", "A short but complete answer about GPUs.")],
                      cfg, max_seq_len=2048)
        self.assertEqual(rep.over_limit, 0)
        self.assertFalse(any("TRUNCATED" in w for w in rep.warnings))

    def test_leakage_between_splits_is_caught(self):
        shared = "How do I free a GPU allocation?"
        train = [self._rec(shared, "Scale to zero replicas.", url="file:///a")]
        evalset = [self._rec(shared, "Scale the deployment to zero.", url="file:///b")]
        count, examples = check_leakage(train, evalset)
        self.assertEqual(count, 1)
        self.assertTrue(examples)

    def test_disjoint_splits_report_no_leakage(self):
        train = [self._rec("How do I free a GPU?", "Scale to zero.")]
        evalset = [self._rec("Where are model artifacts stored?", "In the lakehouse bucket.")]
        self.assertEqual(check_leakage(train, evalset)[0], 0)
        self.assertEqual(check_leakage([], evalset)[0], 0)   # empty split is not a leak

    def test_degenerate_rows_are_flagged(self):
        bad = find_degenerate([
            self._rec("Empty?", ""),
            self._rec("Cut off?", "The procedure continues with the next step and then"),
            self._rec("Fenced?", "Run this:\n```\nkubectl get pods\n"),
            self._rec("Fine?", "This is a complete, properly terminated answer."),
        ])
        self.assertEqual(bad.get("empty_output"), 1)
        self.assertEqual(bad.get("answer_looks_truncated"), 2)  # the empty-ish + fenced
        self.assertEqual(bad.get("unbalanced_code_fence"), 1)

    def test_small_dataset_warns_about_fine_tuning_viability(self):
        cfg = load_config()
        rep = analyze([self._rec(f"Q{i}?", f"A complete answer number {i}.") for i in range(5)],
                      cfg)
        self.assertTrue(any("fine-tune" in w for w in rep.warnings))

    def test_card_contains_provenance_and_limitations(self):
        cfg = load_config()
        rep = analyze([self._rec("How do I free a GPU?", "Scale the endpoint to zero replicas.")],
                      cfg)
        card = dataset_card(rep, cfg)
        self.assertIn("# Dataset Card", card)
        self.assertIn("Limitations and risks", card)
        self.assertIn("licensing is not verified", card)
        self.assertIn(cfg.llm.model, card)
        self.assertIn("Synthetic", card)

    def test_empty_dataset_is_reported_not_crashed(self):
        rep = analyze([], load_config())
        self.assertEqual(rep.total, 0)
        self.assertTrue(rep.warnings)


class TestWebUI(unittest.TestCase):
    """The upload and config-rewrite paths take untrusted input from a browser."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        (self.root / "docs").mkdir()
        (self.root / "runbooks").mkdir()

    def tearDown(self):
        self.dir.cleanup()

    def test_normal_upload_resolves_inside_the_corpus(self):
        p = safe_upload_path(self.root, "docs", "manual.pdf")
        self.assertTrue(str(p).startswith(str(self.root.resolve())))
        self.assertEqual(p.name, "manual.pdf")

    def test_path_traversal_is_stripped(self):
        for evil in ("../../../etc/passwd.txt", "..\\..\\windows\\system32\\x.txt",
                     "sub/dir/notes.md", "/absolute/path/notes.md"):
            p = safe_upload_path(self.root, "docs", evil)
            self.assertTrue(str(p.resolve()).startswith(str(self.root.resolve())), evil)
            self.assertNotIn("..", p.parts)

    def test_unknown_extensions_are_refused(self):
        for bad in ("payload.exe", "script.bat", "lib.dll", "noextension"):
            with self.assertRaises(ValueError, msg=bad):
                safe_upload_path(self.root, "docs", bad)

    def test_only_known_folders_are_allowed(self):
        with self.assertRaises(ValueError):
            safe_upload_path(self.root, "../secrets", "a.pdf")
        with self.assertRaises(ValueError):
            safe_upload_path(self.root, "anything", "a.pdf")

    def test_illegal_filename_characters_are_replaced(self):
        p = safe_upload_path(self.root, "docs", 'we:ird*name?.pdf')
        self.assertNotRegex(p.name, r'[<>:"|?*]')
        self.assertTrue(p.name.endswith(".pdf"))

    def test_keyword_rewrite_preserves_the_rest_of_the_file(self):
        cfg_path = self.root / "config.toml"
        cfg_path.write_text(
            '[project]\nname = "x"\n\n'
            '[sources.keywords]\nenabled = true\nengine = "searxng"\n'
            'terms = [\n  "old one",\n]\n\n'
            '[export]\nout_dir = "exports"\n',
            encoding="utf-8",
        )
        self.assertTrue(update_keywords(cfg_path, ["new one", "another"]))

        text = cfg_path.read_text(encoding="utf-8")
        self.assertIn('"new one"', text)
        self.assertIn('"another"', text)
        self.assertNotIn("old one", text)
        # Untouched neighbours
        self.assertIn('engine = "searxng"', text)
        self.assertIn('out_dir = "exports"', text)

        import tomllib
        parsed = tomllib.loads(text)          # must still be valid TOML
        self.assertEqual(parsed["sources"]["keywords"]["terms"], ["new one", "another"])

    def test_keyword_rewrite_handles_an_empty_list(self):
        cfg_path = self.root / "config.toml"
        cfg_path.write_text('[sources.keywords]\nterms = ["a"]\n', encoding="utf-8")
        self.assertTrue(update_keywords(cfg_path, []))
        import tomllib
        self.assertEqual(tomllib.loads(cfg_path.read_text(encoding="utf-8"))
                         ["sources"]["keywords"]["terms"], [])

    def test_keyword_rewrite_reports_failure_when_the_block_is_missing(self):
        cfg_path = self.root / "config.toml"
        cfg_path.write_text('[project]\nname = "x"\n', encoding="utf-8")
        self.assertFalse(update_keywords(cfg_path, ["a"]))

    def test_project_rewrite_updates_name_and_description(self):
        cfg_path = self.root / "config.toml"
        cfg_path.write_text('[project]\nname = "old"\ndescription = "old desc"\n',
                            encoding="utf-8")
        self.assertTrue(update_project(cfg_path, "new", "new desc"))
        import tomllib
        parsed = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
        self.assertEqual(parsed["project"]["name"], "new")
        self.assertEqual(parsed["project"]["description"], "new desc")

    SAMPLE = '''# leading comment
[llm]
provider     = "ollama"          # trailing comment
base_url     = "http://localhost:11434"
model        = "qwen2.5-coder:7b"
temperature  = 0.2

[llm.embeddings]
enabled = true
model   = "nomic-embed-text"

[generation]
kinds           = ["qa", "instruction"]
pairs_per_chunk = 3

[agent]
objective = """
multi-line
objective text
"""
enabled = true

[[sources.confluence]]
enabled   = false
spaces    = ["A", "B"]
max_pages = 25
'''

    def _sample(self) -> Path:
        p = self.root / "config.toml"
        p.write_text(self.SAMPLE, encoding="utf-8")
        return p

    def test_toml_update_replaces_values_and_keeps_comments(self):
        import tomllib
        p = self._sample()
        changed = update_toml(p, "llm", {"model": "gemma4:latest", "temperature": 0.5})
        self.assertEqual(set(changed), {"model", "temperature"})

        text = p.read_text(encoding="utf-8")
        self.assertIn("# leading comment", text)
        self.assertIn("# trailing comment", text)     # inline comment survives
        parsed = tomllib.loads(text)
        self.assertEqual(parsed["llm"]["model"], "gemma4:latest")
        self.assertEqual(parsed["llm"]["temperature"], 0.5)
        self.assertEqual(parsed["llm"]["base_url"], "http://localhost:11434")

    def test_toml_update_targets_the_right_table(self):
        import tomllib
        p = self._sample()
        # Both [llm] and [llm.embeddings] have a `model` key.
        update_toml(p, "llm.embeddings", {"model": "mxbai-embed-large"})
        parsed = tomllib.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(parsed["llm"]["embeddings"]["model"], "mxbai-embed-large")
        self.assertEqual(parsed["llm"]["model"], "qwen2.5-coder:7b")   # untouched

    def test_toml_update_handles_lists_and_bools(self):
        import tomllib
        p = self._sample()
        update_toml(p, "generation", {"kinds": ["qa", "glossary", "troubleshooting"]})
        update_toml(p, "sources.confluence", {"enabled": True, "spaces": ["OPS"]})
        parsed = tomllib.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(parsed["generation"]["kinds"], ["qa", "glossary", "troubleshooting"])
        self.assertTrue(parsed["sources"]["confluence"][0]["enabled"])
        self.assertEqual(parsed["sources"]["confluence"][0]["spaces"], ["OPS"])

    def test_toml_update_does_not_corrupt_multiline_strings(self):
        import tomllib
        p = self._sample()
        update_toml(p, "agent", {"enabled": False})
        parsed = tomllib.loads(p.read_text(encoding="utf-8"))
        self.assertIn("multi-line", parsed["agent"]["objective"])
        self.assertFalse(parsed["agent"]["enabled"])

    def test_toml_update_appends_a_missing_key(self):
        import tomllib
        p = self._sample()
        update_toml(p, "sources.confluence", {"updated_since": "-30d"})
        parsed = tomllib.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(parsed["sources"]["confluence"][0]["updated_since"], "-30d")

    def test_toml_update_is_a_noop_when_nothing_differs(self):
        p = self._sample()
        before = p.read_text(encoding="utf-8")
        self.assertEqual(update_toml(p, "llm", {"model": "qwen2.5-coder:7b"}), [])
        self.assertEqual(p.read_text(encoding="utf-8"), before)

    def test_toml_update_rejects_a_missing_section(self):
        p = self._sample()
        with self.assertRaises(ValueError):
            update_toml(p, "nope", {"a": 1})

    def test_toml_update_escapes_quotes_rather_than_breaking_the_file(self):
        import tomllib
        p = self._sample()
        update_toml(p, "llm", {"model": 'weird"name'})
        parsed = tomllib.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(parsed["llm"]["model"], 'weird"name')

    def test_env_update_preserves_comments_and_replaces_in_place(self):
        p = self.root / ".env"
        p.write_text("# a comment\nCONFLUENCE_BASE_URL=\nCONFLUENCE_TOKEN=old\n", encoding="utf-8")
        changed = update_env(p, {"CONFLUENCE_BASE_URL": "https://x.atlassian.net/wiki",
                                 "CONFLUENCE_TOKEN": "new"})
        self.assertEqual(set(changed), {"CONFLUENCE_BASE_URL", "CONFLUENCE_TOKEN"})
        text = p.read_text(encoding="utf-8")
        self.assertIn("# a comment", text)
        self.assertIn("CONFLUENCE_BASE_URL=https://x.atlassian.net/wiki", text)
        self.assertIn("CONFLUENCE_TOKEN=new", text)
        self.assertNotIn("=old", text)

    def test_env_update_appends_unknown_keys(self):
        p = self.root / ".env"
        p.write_text("EXISTING=1\n", encoding="utf-8")
        update_env(p, {"BRAND_NEW": "2"})
        self.assertIn("BRAND_NEW=2", p.read_text(encoding="utf-8"))

    def test_secret_masking_never_reveals_the_middle(self):
        self.assertEqual(mask_secret(""), "")
        self.assertNotIn("SECRET", mask_secret("ATATT-SUPER-SECRET-VALUE"))
        self.assertTrue(mask_secret("ATATT-SUPER-SECRET-VALUE").startswith("ATAT"))
        self.assertEqual(mask_secret("short"), "•" * 5)

    def test_ui_file_exists(self):
        self.assertTrue((Path(__file__).resolve().parent.parent
                         / "datagen" / "webui" / "index.html").is_file())


class TestLLMConfig(unittest.TestCase):
    def test_local_addresses_are_recognised(self):
        for url in ("http://localhost:11434", "http://127.0.0.1:1234",
                    "http://192.168.1.50:11434", "http://10.0.0.4:8000",
                    "http://gpubox.local:11434"):
            self.assertTrue(LLMConfig(base_url=url).is_local, url)

    def test_remote_addresses_are_flagged(self):
        for url in ("https://api.openai.com", "https://api.anthropic.com",
                    "http://203.0.113.5:11434"):
            self.assertFalse(LLMConfig(base_url=url).is_local, url)


if __name__ == "__main__":
    unittest.main(verbosity=2)
