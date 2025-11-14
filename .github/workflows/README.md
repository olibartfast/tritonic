# CI/CD Workflows

This directory contains GitHub Actions workflows for continuous integration and deployment.

## Workflows

### 🔨 CI Workflow (`ci.yml`)
**Triggers:** Push and Pull Requests to `master`, `main`, or `develop` branches

**Jobs:**
- **build-and-test**: Builds the C++ project and runs unit tests
  - Caches Triton client libraries for faster builds
  - Installs all required dependencies (OpenCV, RapidJSON, etc.)
  - Runs CMake build with testing enabled
  - Executes unit tests with CTest
  
- **docker-build**: Builds Docker images
  - Builds both production (`Dockerfile`) and development (`Dockerfile.dev`) images
  - Uses Docker layer caching for efficiency
  
- **code-quality**: Runs code quality checks
  - Executes pre-commit hooks on all files

### 🔍 Pre-commit Workflow (`pre-commit.yml`)
**Triggers:** Pull Requests to `master`, `main`, or `develop` branches

Validates code formatting and linting using pre-commit hooks. Caches hooks for faster execution.

### 🐋 Docker Publish Workflow (`docker-publish.yml`)
**Triggers:** 
- Release publications
- Tags matching `v*.*.*` pattern

**Features:**
- Publishes images to GitHub Container Registry (ghcr.io)
- Creates multiple tags: version, major.minor, major, sha, latest
- Builds both production and development images
- Uses semantic versioning

**Published Images:**
- `ghcr.io/<owner>/tritonic:latest` - Latest stable release
- `ghcr.io/<owner>/tritonic:<version>` - Specific version
- `ghcr.io/<owner>/tritonic:dev` - Latest development build

### 🔒 Security Workflow (`security.yml`)
**Triggers:** 
- Push to `master` or `main` branches
- Pull Requests
- Weekly schedule (Mondays at 00:00 UTC)

**Jobs:**
- **codeql**: Static code analysis for C++ and Python
- **docker-scan**: Scans Docker images for vulnerabilities using Trivy
- **dependency-review**: Reviews dependencies in pull requests

## Setup Instructions

### Prerequisites

1. **Enable GitHub Actions** in your repository settings
2. **Enable GitHub Packages** for Docker image publishing
3. **Configure branch protection rules** (recommended)

### Secrets Configuration

No additional secrets are required for basic operation. The workflows use `GITHUB_TOKEN` which is automatically provided.

### Optional: Custom Docker Registry

To publish to Docker Hub or another registry, modify `docker-publish.yml`:

```yaml
env:
  REGISTRY: docker.io  # Change registry
  IMAGE_NAME: <username>/tritonic  # Change image name
```

And add these secrets to your repository:
- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`

Update the login step:
```yaml
- name: Log in to Docker Hub
  uses: docker/login-action@v3
  with:
    registry: docker.io
    username: ${{ secrets.DOCKER_USERNAME }}
    password: ${{ secrets.DOCKER_PASSWORD }}
```

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

### Creating a Release

To trigger the Docker publish workflow:

1. **Create a tag:**
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```

2. **Or create a GitHub Release:**
   - Go to Releases → "Draft a new release"
   - Choose or create a tag (e.g., `v1.0.0`)
   - Fill in release notes
   - Publish release

The workflow will automatically build and publish Docker images.

## Workflow Status Badges

Add these badges to your main README.md:

```markdown
[![CI](https://github.com/<owner>/tritonic/actions/workflows/ci.yml/badge.svg)](https://github.com/<owner>/tritonic/actions/workflows/ci.yml)
[![Security Scan](https://github.com/<owner>/tritonic/actions/workflows/security.yml/badge.svg)](https://github.com/<owner>/tritonic/actions/workflows/security.yml)
[![Docker Publish](https://github.com/<owner>/tritonic/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/<owner>/tritonic/actions/workflows/docker-publish.yml)
```

Replace `<owner>` with your GitHub username or organization name.

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

### Platform Support

Currently, Docker images are built for `linux/amd64`. To add ARM support:

```yaml
platforms: linux/amd64,linux/arm64
```

Note: This will increase build time significantly.

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

### Docker Build Failures

**Issue:** Image too large

**Solution:** Use multi-stage builds and remove unnecessary files in Dockerfile.

## Maintenance

- **Weekly**: Review security scan results
- **Monthly**: Update workflow action versions
- **Per release**: Verify Docker images publish correctly

## Support

For issues with CI/CD workflows, please open an issue with:
- Workflow run URL
- Error messages
- Expected vs actual behavior
