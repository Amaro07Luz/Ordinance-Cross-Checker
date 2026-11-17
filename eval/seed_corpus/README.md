# Seed corpus (SYNTHETIC)

These files are made-up ordinance excerpts written in the style of each
jurisdiction's code. The numbers are placeholders, not the real regulations.
They exist so the API, eval harness and CI can run end to end without
downloading real documents. Folder names are the slugs Part B's
`discover_raw_files()` maps to Part A's `SourceRegistry`.

Replace with (or add alongside) the real documents Part A collects, then
update `eval/queries.jsonl` expectations to match the real section numbers.
