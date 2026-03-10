# Python Setup Guide for FIREworks

The error "Python was not found" is due to Windows app execution aliases redirecting to the Microsoft Store. Here are solutions:

## Option 1: Disable Windows Python Alias (Recommended)

1. Open Settings → Apps → Advanced app settings
2. Scroll to "App execution aliases"
3. Find "python.exe" and "python3.exe"
4. Toggle both OFF

Then try: `python -m unittest discover tests/`

## Option 2: Use Full Python Path

Find your Python installation:

```powershell
Get-Command python.exe | Select-Object -ExpandProperty Definition
```

Then run tests with the full path, or create a batch file wrapper.

## Option 3: Install Python Fresh

Download Python from https://www.python.org/ and install with:
- ✅ "Add Python to PATH" checkbox enabled
- Uncheck "App execution aliases" option

## Option 4: Use UV Package Manager (Recommended for Projects)

UV is a fast Python package manager. Install from: https://docs.astral.sh/uv/

```bash
uv sync  # Install dependencies
uv run python -m pytest tests/
```

## Option 5: Use Docker

```bash
docker run --rm -v ${PWD}:/work -w /work python:3.11 python -m pytest tests/
```

## Running Tests Once Python Works

```bash
# Run all tests
python -m unittest discover tests/ -v

# Run specific test file
python -m unittest tests.test_strategies.test_mc_strategy -v

# Or use the provided runner script
python run_tests.py
```

## Install Project in Development Mode

Once Python is working:

```bash
pip install -e .
# or with uv:
uv pip install -e .
```

This allows you to import `fireworks` from anywhere in the project.
