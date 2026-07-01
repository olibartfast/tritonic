#!/bin/bash
# Format all C++ source files in the repository
# Run this before committing to ensure CI format checks pass

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🔍 Formatting all C++ files in the repository..."
echo ""

# Check if clang-format is installed
if ! command -v clang-format &> /dev/null; then
    echo "❌ Error: clang-format not found!"
    echo ""
    echo "Please install clang-format:"
    echo "  Ubuntu/Debian: sudo apt-get install clang-format"
    echo "  macOS: brew install clang-format"
    echo "  Arch Linux: sudo pacman -S clang"
    exit 1
fi

CLANG_FORMAT_VERSION=$(clang-format --version | head -n1)
echo "Using: $CLANG_FORMAT_VERSION"
echo ""

# Find and format all C++ files
cd "$REPO_ROOT"

FILES_FORMATTED=0
for dir in include src tests; do
    if [ -d "$dir" ]; then
        echo "Processing: $dir/"
        while IFS= read -r -d '' file; do
            echo "  Formatting: $file"
            clang-format -i "$file"
            ((++FILES_FORMATTED))
        done < <(find "$dir" -type f \( -name '*.cpp' -o -name '*.hpp' \) -print0)
    fi
done

echo ""
echo "✅ Formatted $FILES_FORMATTED files successfully!"
echo ""
echo "💡 Next steps:"
echo "  1. Review the changes: git diff"
echo "  2. Stage the changes: git add ."
echo "  3. Commit: git commit -m 'style: apply clang-format'"
