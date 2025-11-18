# CI/CD Workflows

This directory contains GitHub Actions workflows for continuous integration and code quality.

## Workflows

### 🔨 CI Workflow (`ci.yml`)
**Triggers:** 
- Push to `master`, `main`, or `develop` branches
- Pull Requests to `master`, `main`, or `develop` branches  
- Weekly schedule (Mondays at 00:00 UTC for security scans)
- Ignores: Documentation files (`.md`), `docs/**`

**Jobs:**

#### 1. **build-and-test** (Ubuntu 24.04)
Builds the C++ project and runs unit tests.
- Caches Triton client libraries (r25.06) for faster builds
- Extracts Triton libraries from Docker SDK image if not cached
- Installs system dependencies: CMake, RapidJSON, libcurl, Protobuf, GTest, GMock
- Installs OpenCV development libraries
- Configures CMake with testing enabled
- Builds project using all CPU cores (`-j$(nproc)`)
- Runs unit tests directly via test executable or CTest

#### 2. **codeql** (Security Analysis)
Static code analysis for security vulnerabilities.
- Analyzes both C++ and Python code
- Attempts to build C++ project for comprehensive analysis
- Uploads results to GitHub Security tab
- Runs on: push, PR, and weekly schedule

#### 3. **dependency-review** (PR only)
Reviews dependency changes in pull requests.
- Checks for security vulnerabilities in dependencies
- Fails on moderate or higher severity issues
- Automatically comments on PRs with findings

#### 4. **code-quality**
Runs code formatting and linting checks.
- Executes pre-commit hooks on all files
- Validates code style consistency
- Non-blocking (continues on failure)

#### 5. **code-review** (PR only)
Automated code review for pull requests.
- **AI Code Review**: Uses OpenAI to analyze code changes (requires `OPENAI_API_KEY` secret)
  - Excludes documentation and configuration files
  - Reviews up to 10 files per PR
- **Complexity Analysis**: Uses lizard to detect complex C++ functions
  - Flags functions with Cyclomatic Complexity > 15
  - Provides refactoring recommendations
- **PR Statistics**: Generates comprehensive PR metrics
  - Files changed, lines added/deleted
  - File type breakdown
  - Warnings for large PRs (>20 files or >500 changes)

## Setup Instructions

### Prerequisites

1. **Enable GitHub Actions** in your repository settings
2. **Configure branch protection rules** (recommended)
3. **Docker installed** (for local Triton library extraction)

### Secrets Configuration

**Required for basic operation:**
- `GITHUB_TOKEN` - Automatically provided by GitHub Actions

**Optional secrets for enhanced features:**
- `OPENAI_API_KEY` - Enables AI-powered code review in PRs
  - Create at: https://platform.openai.com/api-keys
  - Add to: Repository Settings → Secrets and variables → Actions → New repository secret
  - Without this secret, AI code review will be skipped (other review features still work)

## Usage

### Running Tests Locally

Before pushing, you can run tests locally:

```bash
# Build with testing enabled
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON ..
cmake --build .

# Run tests
ctest --output-on-failure
```

### Pre-commit Hooks

Install pre-commit hooks locally:

```bash
./scripts/setup/pre_commit_setup.sh
```

This ensures code quality checks run before each commit.

### Testing Pull Requests

When you create a PR, the following automated checks will run:

1. **Build and Test** - Ensures code compiles and tests pass
2. **CodeQL Analysis** - Security vulnerability scanning  
3. **Dependency Review** - Checks for vulnerable dependencies
4. **Code Quality** - Pre-commit hooks validation
5. **Code Review** - AI review, complexity analysis, and PR statistics

All results appear in the PR checks section and as comments on the PR.

## Workflow Status Badges

The CI workflow badge is already in the main README:

```markdown
[![CI](https://github.com/olibartfast/tritonic/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/olibartfast/tritonic/actions/workflows/ci.yml)
```

This badge shows the status of builds, tests, and security scans on the master branch.

## Customization

### Adjusting Build Options

Modify `ci.yml` to enable/disable build options:

```yaml
- name: Configure CMake
  run: |
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_TESTING=ON \
          -DWITH_SHOW_FRAME=ON \
          -DWITH_WRITE_FRAME=ON \
          ..
```

### Changing Test Timeout

Edit the test timeout in `tests/CMakeLists.txt`:

```cmake
set_tests_properties(UnitTests PROPERTIES
    TIMEOUT 600  # Increase to 10 minutes
)
```

### Enabling AI Code Review

To enable AI-powered code review:

1. Create an OpenAI API key at https://platform.openai.com/api-keys
2. Add it as a repository secret:
   - Go to: Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `OPENAI_API_KEY`
   - Value: Your OpenAI API key
3. The AI review will automatically activate on the next PR

Without this key, PRs will still get complexity analysis and statistics.

## Troubleshooting

### Build Failures

**Issue:** Triton client libraries not found

**Solution:** The workflow caches Triton libraries. If extraction fails, clear the cache:
- Go to Actions → Caches
- Delete the `triton-client-*` cache
- Re-run the workflow

**Issue:** OpenCV not found

**Solution:** Ensure `libopencv-dev` and related packages are in the dependency installation step.

### Test Failures

**Issue:** Tests timeout

**Solution:** Increase the timeout in `tests/CMakeLists.txt` or reduce test complexity.

### AI Code Review Not Working

**Issue:** AI review step is skipped

**Solution:** Ensure `OPENAI_API_KEY` secret is configured in repository settings.

**Issue:** AI review fails with authentication error

**Solution:** Verify your OpenAI API key is valid and has sufficient credits.

## Maintenance

- **Weekly**: Review security scan results
- **Monthly**: Update workflow action versions
- **Per release**: Verify Docker images publish correctly

## Support

For issues with CI/CD workflows, please open an issue with:
- Workflow run URL
- Error messages
- Expected vs actual behavior
