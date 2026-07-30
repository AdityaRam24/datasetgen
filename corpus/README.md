# corpus/ — put your source material here

Anything you drop in these folders is picked up on the next `datagen build`.
Subfolders are walked, so organise them however you like.

## corpus/docs/  — reference material

PDFs, Word, PowerPoint, Excel, Markdown, text, HTML, CSV, JSON.

    corpus/docs/
      vendor-manuals/     product-guide.pdf
      internal/           architecture.docx
      exports/            metrics.xlsx

Scanned PDFs will NOT work — there is no OCR. If a PDF is a photo of a page,
run it through OCR first (e.g. ocrmypdf) and then drop it here.

## corpus/runbooks/  — procedures

Markdown, text, PDF, Word.

Files here go through the runbook parser instead of the plain document parser:
it recognises symptom / cause / preconditions / steps / verification /
rollback / escalation sections, pulls out ordered steps and referenced
commands, and preserves their sequence. That produces much better
troubleshooting examples than treating the same file as flat prose.

See `mlis-endpoint-unavailable.md` for the structure it reads best. You do not
have to match it exactly — it falls back gracefully — but the closer you are,
the better the output.

## What happens next

    python -m datagen build       # parse -> chunk -> dedupe -> generate -> gate
    python -m datagen inspect     # see the rows it produced
    python -m datagen analyze     # is it trainable?

`build` is incremental: unchanged files are skipped by content hash, so
re-running after adding a few documents only processes the new ones.

## Notes

- The two sample files here are examples. Delete them once you have your own.
- File formats are configured in `config.toml` under `[[sources.files]]` and
  `[[sources.runbooks]]` — add a glob there if you need another extension.
- Confidential material: `corpus/` is currently tracked by git. Add `corpus/`
  to `.gitignore` before committing anything you would not publish.
