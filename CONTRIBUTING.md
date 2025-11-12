# Contributing to Humanitarian Evaluation AI Research Pipeline

Thank you for your interest in contributing to the Humanitarian Evaluation AI Research Pipeline! This document provides guidelines and instructions for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Making Changes](#making-changes)
- [Commit Guidelines](#commit-guidelines)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Code Style](#code-style)

## 🤝 Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please treat all community members with respect and create a welcoming environment for everyone.

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing to ensure a positive and inclusive community experience.

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- A GitHub account

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/humanitarian-evaluation-ai-research.git
   cd humanitarian-evaluation-ai-research
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/ORIGINAL-OWNER/humanitarian-evaluation-ai-research.git
   ```

## 💻 Development Setup

See main README.

## 🔧 Pre-commit Hooks

We use pre-commit hooks to ensure code quality and consistency. **This is required for all contributors.**

### Install Pre-commit

```bash
# Activate your virtual environment first
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate    # On Windows

# Install pre-commit package (if not already installed)
pip install pre-commit

# Install the git hook scripts
pre-commit install
```

### What Pre-commit Does

Our pre-commit configuration (`.pre-commit-config.yaml`) automatically:

- **Code Formatting**: Runs `black` to format Python code
- **Import Sorting**: Runs `isort` to organize imports
- **Linting**: Runs `flake8` for code quality checks
- **Type Checking**: Runs `mypy` for static type analysis
- **File Checks**: Removes trailing whitespace, fixes end-of-file formatting
- **YAML/JSON Validation**: Checks syntax of configuration files
- **Merge Conflict Detection**: Prevents committing unresolved conflicts
- **Large File Prevention**: Blocks accidentally committed large files

### Running Pre-commit Manually

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run

# Update pre-commit hooks to latest versions
pre-commit autoupdate

# Skip pre-commit for emergency commits (not recommended - violates project rules)
# Note: User rules prohibit --no-verify flag
git commit -m "your message"  # Always runs pre-commit hooks
```

## 🛠️ Making Changes

### Development Workflow

1. **Create a feature branch:**
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes:**
   - Write code following our [Code Style](#code-style)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes:**
   ```bash
   # Activate virtual environment
   source .venv/bin/activate  # macOS/Linux

   # Run tests
   pytest

   # Run with coverage
   pytest --cov=pipeline --cov-report=html --cov-report=term-missing

   # Run pre-commit checks
   pre-commit run --all-files
   ```

4. **Commit your changes** (see [Commit Guidelines](#commit-guidelines))

5. **Push and create PR:**
   ```bash
   git push origin feat/your-feature-name
   ```

### Types of Contributions

- **Bug Fixes**: Fix existing issues
- **Features**: Add new functionality
- **Documentation**: Improve docs, examples, or comments
- **Tests**: Add or improve test coverage
- **Performance**: Optimize existing code
- **Refactoring**: Improve code structure without changing functionality

## 📝 Commit Guidelines

We use [Conventional Commits](https://www.conventionalcommits.org/) format:

### Commit Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Commit Types

- **feat:** New features
- **fix:** Bug fixes
- **docs:** Documentation changes
- **style:** Code style changes (formatting, etc.)
- **refactor:** Code refactoring
- **test:** Adding or modifying tests
- **chore:** Build process or auxiliary tool changes
- **perf:** Performance improvements
- **ci:** CI/CD pipeline changes
- **build:** Build system or dependencies

### Examples

```bash
feat: add language detection to parse pipeline
fix: resolve PDF parsing error for large documents
docs: update README with metadata preparation examples
test: add unit tests for summarization module
chore: update dependencies to latest versions
```

### Commit Best Practices

- Keep commits atomic (one logical change per commit)
- Write clear, descriptive commit messages
- Reference issues when applicable: `fixes #123`
- Use imperative mood: "add feature" not "added feature"

## 🧪 Testing

### Running Tests

```bash
# Activate virtual environment
source .venv/bin/activate  # macOS/Linux

# Run all tests
pytest

# Run with coverage report
pytest --cov=pipeline --cov-report=html --cov-report=term-missing

# Run specific test file
pytest tests/test_basic.py

# Run with verbose output
pytest -v
```

### Test Categories

- **Unit Tests** (`tests/`): Test individual pipeline components

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names: `test_parse_pdf_with_valid_document`
- Mock external API calls (e.g., HuggingFace API)
- Test both success and error scenarios
- Aim for high test coverage (>90%)

### Manual Testing

Test the pipeline manually:

```bash
# Activate virtual environment
source .venv/bin/activate

# Test parsing
python pipeline/parse.py --metadata data/pdf_metadata_sample.xlsx

# Test summarization
python pipeline/summarize.py --metadata data/pdf_metadata_sample.xlsx

# Test statistics generation
python pipeline/stats.py --metadata data/pdf_metadata_sample.xlsx
```

## 🔍 Pull Request Process

### Before Submitting

1. **Update your branch:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks:**
   ```bash
   source .venv/bin/activate  # Activate virtual environment
   pytest
   pre-commit run --all-files
   ```

3. **Update documentation** if needed

### PR Guidelines

- **Title**: Use conventional commit format
- **Description**: Clearly explain what and why
- **Link Issues**: Reference related issues
- **Screenshots**: Include for UI changes
- **Breaking Changes**: Clearly document any breaking changes

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other: ___

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review of code completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and linting
2. **Peer Review**: At least one maintainer reviews the code
3. **Feedback**: Address any requested changes
4. **Approval**: Maintainer approves and merges

## 🐛 Issue Guidelines

### Before Creating an Issue

1. **Search existing issues** to avoid duplicates
2. **Check documentation** for common solutions
3. **Test with latest version**

### Bug Reports

Include:
- **Environment**: OS, Python version, package versions
- **Steps to reproduce** the issue
- **Expected behavior**
- **Actual behavior**
- **Error messages** (full stack trace)
- **Minimal code example**
- **Sample metadata file** (if related to parsing/summarization)

### Feature Requests

Include:
- **Clear description** of the feature
- **Use case** and motivation
- **Proposed solution** (if you have one)
- **Alternatives considered**

## 🎨 Code Style

### Python Style Guide

- **Formatter**: Black (line length: 88 characters)
- **Import Sorting**: isort (Black-compatible profile)
- **Linting**: flake8 with extensions for complexity and security
- **Type Checking**: mypy (strict mode)
- **Type Hints**: Use type hints for function signatures
- **Docstrings**: Google style docstrings

### Code Quality Requirements

1. **Type Hints**: All functions must have type hints
2. **Docstrings**: All public functions and classes must have docstrings
3. **Test Coverage**: Minimum 90% test coverage for new code
4. **Security**: No security vulnerabilities (checked by Bandit)
5. **Complexity**: Keep cyclomatic complexity under 10

### Code Quality Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Format code
black pipeline/ tests/
isort pipeline/ tests/

# Check code quality
flake8 pipeline/ tests/
mypy pipeline/

# Run all quality checks
pre-commit run --all-files
```

## 🚀 Adding New Features

### Process

1. **New Pipeline Scripts**: Add new pipeline components in the `pipeline/` directory
2. **Include proper type hints and documentation** for all functions
3. **Add corresponding tests** in `tests/`
4. **Update the README** with documentation for your feature
5. **Update metadata columns** if your feature adds new data to the Excel file

### Extending the Pipeline

New pipeline stages should:
- Read from and write to the metadata Excel file
- Follow the existing pattern of parse → summarize → stats
- Include error handling and logging
- Support command-line arguments for flexibility

## 🔄 Release Process

For maintainers:

1. **Version Bump**: Update version in `pyproject.toml`
2. **Changelog**: Update release notes
3. **Tag Release**: Create git tag
4. **GitHub Release**: Create release with notes
5. **PyPI**: Automated via GitHub Actions

## 📞 Getting Help

- **Documentation**: Check the README and code comments
- **Issues**: Create a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **UNEG Resources**: Consult [UNEG Evaluation Reports](https://www.unevaluation.org/)

## 🙏 Recognition

Contributors will be recognized in:
- GitHub contributors list
- Release notes for significant contributions
- Project acknowledgments

Thank you for contributing to the Humanitarian Evaluation AI Research Pipeline! 🎉
