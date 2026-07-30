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

from datagen.chunking import chunk_document                      # noqa: E402
from datagen.config import ChunkingConfig, LLMConfig, QualityConfig, load_config  # noqa: E402
from datagen.connectors.parsers import parse_html, _postprocess_pdf  # noqa: E402
from datagen.connectors.runbooks import normalize, parse_runbook  # noqa: E402
from datagen.dedupe import Deduper, hamming, simhash             # noqa: E402
from datagen.exporters import (                                   # noqa: E402
    build_glossary, glossary_markdown, split_by_document, to_alpaca, to_chatml,
)
from datagen.generators import _extractive_glossary, is_useful_term  # noqa: E402
from datagen.models import Chunk, Document, Record               # noqa: E402
from datagen.quality import grounding_score, heuristic_check     # noqa: E402
from datagen.state import StateStore, _to_signed64, _to_unsigned64  # noqa: E402
from datagen.util import clean_text, extract_json, tokenize      # noqa: E402


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
