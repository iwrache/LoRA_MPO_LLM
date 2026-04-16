# Contributing to MPO Compression

## Development Setup

```bash
# Clone and install
git clone https://github.com/zhc1212/MPO_Compression.git
cd MPO_Compression
pip install -e ".[dev]"
pre-commit install
```

## Running Tests

```bash
# Unit tests (CPU only, no model downloads required)
pytest -m "not integration" -v

# Integration tests (require GPU + model weights)
pytest -m integration --run-integration --model-path <path>

# With coverage
pytest -m "not integration" --cov=mpo_modules --cov-report=html
```

## Code Style

- Python 3.10+
- Linted and formatted with [ruff](https://docs.astral.sh/ruff/)
- Line length: 120 characters
- Run `make format` before committing

## Pull Request Process

1. Create a feature branch from `main`
2. Write tests for new functionality
3. Ensure `make lint` and `make test` pass
4. Submit a PR with a clear description of changes

## Writing Tests

- Unit tests go in `tests/` and must run on CPU without model downloads
- Use `torch.randn` with `torch.Generator().manual_seed(42)` for reproducible fake data
- Use `monkeypatch` to control environment variables
- Mark GPU/model tests with `@pytest.mark.integration`
