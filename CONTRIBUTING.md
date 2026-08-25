# Contributing

## Running tests locally

```sh
python -m pip install -r requirements.txt pytest
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install sentence-transformers numpy
python -m pytest tests/ -v
```

The search stack imports `sentence_transformers` at module scope; tests use a
`FakeModel`, but the import must resolve, hence the CPU torch wheel install.

## CI status

CI (`workflows/ci.yml` — see its header note) runs on every push/PR to
`main`: **Green ✓** = dependency install + full pytest suite passed;
**Red ✗** = open the workflow run in the Actions tab to read the first
failing test; a previously-running commit may show **cancelled ⊗** after a
re-push (superseded run, ignore it).
