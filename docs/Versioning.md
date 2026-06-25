# Versioning and Changelog

## Overview

This project uses two files to track releases:

| File | Purpose |
|------|---------|
| `VERSION` | Single source of truth for the current version (read by CMake) |
| `CHANGELOG.md` | Human-readable history of notable changes per release |

## VERSION file

Contains a single line like `0.2.0-dev`.

- The `-dev` suffix indicates unreleased development work on `develop`.
- CMake reads this file at configure time and strips the suffix to set
  `project(tritonic VERSION X.Y.Z)`.
- Follows [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.

## CHANGELOG.md

Follows the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

Sections per release:
- **Added** — new features
- **Changed** — changes to existing functionality
- **Fixed** — bug fixes
- **Removed** — removed features
- **Deprecated** — features marked for future removal

Unreleased work goes under the `[Unreleased]` heading at the top.

## Gitflow model

| Branch | Role |
|--------|------|
| `develop` | Integration branch for normal work |
| `master` | Release-only; every commit on `master` is a tagged release |
| `release/X.Y.Z` | Short-lived branch used to stage a release |
| `feature/*` | Topic branches merged into `develop` via PR |

## Day-to-day workflow

When merging a PR into `develop`, add a line under `[Unreleased]` in the
appropriate section. Example:

```markdown
## [Unreleased]

### Added
- Support for a new model type
```

## Release workflow

1. **Create a release branch** from `develop`:
   ```
   git checkout -b release/0.2.0 develop
   ```

2. **Update VERSION** — remove the `-dev` suffix:
   ```
   0.2.0
   ```

3. **Update CHANGELOG.md** — rename `[Unreleased]` to the new version with
   today's date, and add a fresh empty `[Unreleased]` section:
   ```markdown
   ## [Unreleased]

   ## [0.2.0] - 2026-06-25

   ### Added
   - ...
   ```
   Update the comparison links at the bottom:
   ```markdown
   [Unreleased]: https://github.com/olibartfast/tritonic/compare/v0.2.0...HEAD
   [0.2.0]: https://github.com/olibartfast/tritonic/compare/v0.1.0...v0.2.0
   ```

4. **Merge into `master`** and tag:
   ```
   git checkout master
   git merge release/0.2.0
   git tag v0.2.0
   git push origin master --tags
   ```

5. **Create the GitHub Release** for the tag — **mandatory, never skip**:
   Extract the release section from `CHANGELOG.md` (everything under
   `## [X.Y.Z]` down to the next `## [` header) and pass it via `--notes`:
   ```
   gh release create v0.2.0 --repo olibartfast/tritonic --title "v0.2.0" \
     --notes "$(sed -n '/^## \[0\.2\.0\]/,/^## \[/{ /^## \[/!p}' CHANGELOG.md | sed '/^$/N;/^\n$/d')"
   ```

   Never use `--generate-notes` — it produces commit-based notes that bypass
   the curated `CHANGELOG.md`. The changelog is the single source of truth.

   Every tag must have a corresponding release. There must never be a tag
   visible on GitHub without a matching release entry. If you discover a
   missing release (tag exists, release does not), create it immediately.

6. **Bump develop** — merge back and set the next dev version:
   ```
   git checkout develop
   git merge release/0.2.0
   ```
   Update `VERSION` to `0.3.0-dev`, commit, push.
