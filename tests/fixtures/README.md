# Landmark fixtures

Fixtures contain normalized landmark coordinates and blendshape scores only. They must never
contain camera pixels, images, video, names, or identifying metadata.

Contributor-recorded fixtures are accepted only when the contributor records themself with
informed consent. Do not submit fixtures of minors or third parties. Review every JSONL file
before committing it, and keep each file at or below 200 KB.

Synthetic fixtures under `synthetic/` are generated from the mathematical face model in
`tests/synth.py`; they do not describe a real person.
