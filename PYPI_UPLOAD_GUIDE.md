# Guide to Upload reg-normalizer to PyPI

Your package is ready to be published! Here's what I've done and what you need to do next:

## What I've Set Up:

✅ **Package Structure Created:**
- `pyproject.toml` - Modern Python package configuration
- `setup.py` - Package setup script
- `MANIFEST.in` - Includes README, requirements, and data files
- `reg_normalizer/__init__.py` - Package entry point with RegionMatcher export

✅ **Code Reorganized:**
- Renamed `src` → `reg_normalizer` (proper package name)
- Moved data files into package: `reg_normalizer/data/interim/regions_etalon_v2.0.yaml`
- Updated file paths in code

✅ **Package Built:**
- Created distribution files in `dist/`:
  - `reg_normalizer-1.0.0-py3-none-any.whl` (wheel)
  - `reg_normalizer-1.0.0.tar.gz` (source distribution)

## What You Need to Do:

### Step 1: Update Package Metadata (IMPORTANT!)

Edit `pyproject.toml` and update:

```toml
[project]
authors = [
    {name = "Your Name", email = "your.email@example.com"}  # <-- Change this!
]
```

### Step 2: Create PyPI Account

1. Go to https://pypi.org/account/register/
2. Create an account and verify your email
3. **Enable 2FA** (required for uploading packages)

### Step 3: Create API Token

1. Go to https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Name: "reg-normalizer-upload"
4. Scope: "Entire account" (or specific to this project after first upload)
5. **Save the token** - it starts with `pypi-`

### Step 4: Upload to PyPI

**Option A: Using Twine (Recommended)**

```bash
# Install twine if not already installed
/usr/bin/python3 -m pip install --upgrade twine

# Upload to PyPI
/usr/bin/python3 -m twine upload dist/*

# When prompted:
# Username: __token__
# Password: <paste your PyPI API token>
```

**Option B: Upload to TestPyPI First (Safer)**

Test your package on TestPyPI before the real PyPI:

```bash
# Create account at https://test.pypi.org/account/register/
# Create API token at https://test.pypi.org/manage/account/token/

# Upload to TestPyPI
/usr/bin/python3 -m twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ reg-normalizer

# If it works, upload to real PyPI:
/usr/bin/python3 -m twine upload dist/*
```

### Step 5: Test Installation

After uploading, test your package:

```bash
# Install from PyPI
pip install reg-normalizer

# Test it works
python3 -c "from reg_normalizer import RegionMatcher; print('Success!')"
```

### Step 6: Update README

After publishing, update your README.md installation instructions:

```markdown
## Installation

```bash
pip install reg-normalizer
```

## Quick Start

```python
from reg_normalizer import RegionMatcher

matcher = RegionMatcher()
match, score = matcher.find_best_match("московск область")
print(f"Match: {match}, Score: {score:.2f}")
```
```

## Important Notes:

1. **Package Name**: `reg-normalizer` (with hyphen) is the PyPI name
2. **Import Name**: `reg_normalizer` (with underscore) is the Python import name
3. **Version**: Currently set to 1.0.0 in pyproject.toml
4. **License**: CC BY-NC-SA 4.0 (non-commercial use)

## Future Updates:

To release a new version:

1. Update version in `pyproject.toml`
2. Commit your changes
3. Clean old builds: `rm -rf dist/ build/ *.egg-info`
4. Build: `/usr/bin/python3 -m build`
5. Upload: `/usr/bin/python3 -m twine upload dist/*`

## Troubleshooting:

**Problem**: "File already exists" error
- **Solution**: You can't re-upload the same version. Increment version in pyproject.toml

**Problem**: Import errors after installation
- **Solution**: Make sure NLTK data is downloaded: `python -m nltk.downloader snowball_data`

**Problem**: Can't find data files
- **Solution**: Already handled! Data files are included in the package

## Next Steps:

1. Consider adding automated tests (pytest)
2. Set up GitHub Actions for CI/CD
3. Add badges to README (PyPI version, downloads, etc.)
4. Create documentation with examples

Good luck with your PyPI upload! 🚀
