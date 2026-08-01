# GitHub Actions workflows

Two workflows live here. This file describes what they actually do; to reproduce
the checks locally see [Local CI Checks](../../docs/guides/Local_CI_Checks.md).

## `ci.yml`

**Triggers:** push to `develop`, and pull requests targeting `develop`. Nothing
else — a pull request against `master` runs no CI at all, so open PRs against
`develop`.

Five independent jobs, all on `ubuntu-latest`. Each is a separate signal; there is
no ordering between them.

| Job | Name | What it runs |
| --- | --- | --- |
| `format-check` | Format Check | `clang-format --dry-run --Werror` over `*.cpp`/`*.hpp` in `include`, `src`, `tests` |
| `clang-tidy` | Clang-Tidy | `clang-tidy -p build --header-filter='^(include\|src)/'` over `src/**/*.cpp` |
| `cppcheck` | Cppcheck | `cppcheck --enable=warning --std=c++20 --error-exitcode=1 -I include src/` |
| `build-and-test` | Build & Test | Configure with `-DBUILD_TESTING=ON`, build, then `ctest --output-on-failure` |
| `build-warnings` | Build with Strict Warnings | Configure with `-DTRITONIC_STRICT_WARNINGS=ON`, build only |

The three heavier jobs cache the Triton client libraries — extracted from the
NVIDIA SDK image on a cache miss — and use `ccache`.

Two details worth knowing before relying on a green run:

- **clang-tidy does not fail on warnings.** It runs without
  `-warnings-as-errors`, so only hard errors turn the job red.
- **`build-warnings` does not build the tests.** It configures without
  `BUILD_TESTING`, so test sources are never compiled under `-Werror`. Configuring
  locally with `-DTRITONIC_STRICT_WARNINGS=ON -DBUILD_TESTING=ON` is therefore a
  stricter check than CI performs, and can fail where CI passes.

## `publish-github-release.yml`

Publishes a GitHub release from the changelog on pushes to `develop` and `master`
and on tags. No build or test steps.

## What CI does not cover

Worth stating explicitly, because a green run is easy to over-read:

- **No GPU runner.** The CUDA DALI plugins under `deploy/**/dali_plugin/` are
  never compiled, the ensembles are never deployed, and neither `benchmarks/` nor
  `integration-tests/` is exercised. All of that is verified by hand.
- **No sanitizers.** There is no ASan/UBSan/TSan job, so memory errors reachable
  only at runtime are not caught.
- **`.cu` files are not format-checked.** The `format-check` glob covers
  `*.cpp`/`*.hpp` under `include`, `src` and `tests` only.
- **cppcheck scans `src/` only** — not `tests/`, `include/` or `deploy/`.
- **CodeQL is configured but not wired up.** `.github/codeql-config.yml` exists;
  no workflow references it.
