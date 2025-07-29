#!/bin/bash

# Pre-commit setup script for TritonIC project
# This script installs and configures pre-commit hooks

set -e

echo "🚀 Setting up pre-commit hooks for TritonIC..."

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "📦 Installing pre-commit..."
    pip install pre-commit
else
    echo "✅ pre-commit is already installed"
fi

# Install the git hook scripts
echo "🔧 Installing git hooks..."
pre-commit install

# Install additional dependencies for C++ tools
echo "🔧 Installing C++ development tools..."

# Check OS and install appropriate packages
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v apt-get &> /dev/null; then
        echo "📦 Installing packages via apt..."
        sudo apt-get update
        sudo apt-get install -y clang-format clang-tidy clang-tools
    elif command -v yum &> /dev/null; then
        echo "📦 Installing packages via yum..."
        sudo yum install -y clang-tools-extra
    elif command -v dnf &> /dev/null; then
        echo "📦 Installing packages via dnf..."
        sudo dnf install -y clang-tools-extra
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if command -v brew &> /dev/null; then
        echo "📦 Installing packages via Homebrew..."
        brew install llvm
        echo "⚠️  You may need to add LLVM to your PATH:"
        echo "   export PATH=\"/usr/local/opt/llvm/bin:\$PATH\""
    fi
else
    echo "⚠️  Please install clang-format and clang-tidy manually for your OS"
fi

# Run pre-commit on all files to set up the environment
echo "🔍 Running pre-commit on all files..."
pre-commit run --all-files

echo "✅ Pre-commit setup complete!"
echo ""
echo "📋 Usage:"
echo "  - pre-commit run --all-files    # Run all hooks on all files"
echo "  - pre-commit run                 # Run hooks on staged files"
echo "  - pre-commit run --hook-type pre-commit  # Run specific hook type"
echo ""
echo "🔧 Available hooks:"
echo "  - clang-format: Code formatting"
echo "  - clang-tidy: C++ linting"
echo "  - shellcheck: Shell script linting"
echo "  - yamllint: YAML linting"
echo "  - markdownlint: Markdown linting"
echo "  - Custom hooks: TODO checks, debug print detection, etc."
echo ""
echo "💡 Tips:"
echo "  - Hooks run automatically on git commit"
echo "  - Use 'git commit --no-verify' to skip hooks (not recommended)"
echo "  - Edit .pre-commit-config.yaml to customize hooks" 