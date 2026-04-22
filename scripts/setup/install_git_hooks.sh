#!/bin/bash
# Install git hooks for automatic code formatting
# This prevents CI format check failures

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HOOKS_DIR="$REPO_ROOT/.git/hooks"

echo "🚀 Installing git hooks for TritonIC..."
echo "Repository: $REPO_ROOT"
echo ""

# Check if clang-format is installed
if ! command -v clang-format &> /dev/null; then
    echo "⚠️  Warning: clang-format not found!"
    echo ""
    echo "Please install clang-format:"
    echo "  Ubuntu/Debian: sudo apt-get install clang-format"
    echo "  macOS: brew install clang-format"
    echo "  Arch Linux: sudo pacman -S clang"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    CLANG_FORMAT_VERSION=$(clang-format --version | head -n1)
    echo "✅ Found: $CLANG_FORMAT_VERSION"
fi

echo ""

# Create pre-commit hook
cat > "$HOOKS_DIR/pre-commit" << 'EOF'
#!/bin/bash
# Pre-commit hook to automatically format C++ code with clang-format
# This prevents format check failures in CI

set -e

# Get list of staged C++ files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(cpp|hpp)$' || true)

if [ -z "$STAGED_FILES" ]; then
    exit 0
fi

# Check if clang-format is installed
if ! command -v clang-format &> /dev/null; then
    echo "⚠️  Warning: clang-format not found. Please install it:"
    echo "   Ubuntu/Debian: sudo apt-get install clang-format"
    echo "   macOS: brew install clang-format"
    echo ""
    echo "Skipping format check..."
    exit 0
fi

echo "🔍 Running clang-format on staged files..."

# Format each staged file
for file in $STAGED_FILES; do
    if [ -f "$file" ]; then
        echo "  Formatting: $file"
        clang-format -i "$file"
        git add "$file"
    fi
done

echo "✅ Code formatting complete!"
exit 0
EOF

chmod +x "$HOOKS_DIR/pre-commit"

echo "✅ Pre-commit hook installed successfully!"
echo ""
echo "📋 What this hook does:"
echo "  • Automatically formats C++ files (.cpp, .hpp) before each commit"
echo "  • Uses the project's .clang-format configuration"
echo "  • Prevents CI format check failures"
echo ""
echo "💡 Usage:"
echo "  • The hook runs automatically on 'git commit'"
echo "  • To bypass (not recommended): git commit --no-verify"
echo "  • To manually format all files: ./scripts/format_all.sh"
echo ""
echo "🎉 You're all set! Happy coding!"
