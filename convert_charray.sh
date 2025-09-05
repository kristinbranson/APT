#!/bin/bash

# Script to convert "charray" to "char" while respecting capitalization
# - "charray" -> "char"
# - "Charray" -> "Char" 
# - "CHARRAY" -> "CHAR"

# Get the directory to search (default to current directory)
SEARCH_DIR="${1:-.}"

echo "Converting charray to char in directory: $SEARCH_DIR"

# Find all .m files and apply the transformations
find "$SEARCH_DIR" -name "*.m" -type f -exec sed -i \
  -e 's/charray/char/g' \
  -e 's/Charray/Char/g' \
  -e 's/CHARRAY/CHAR/g' \
  {} \;

echo "Conversion complete!"
echo ""
echo "Summary of changes made:"

# Show a summary of what was changed
find "$SEARCH_DIR" -name "*.m" -type f -exec grep -l -E "(char|Char|CHAR)" {} \; | head -10 | while read file; do
  echo "Modified: $file"
done

if [ $(find "$SEARCH_DIR" -name "*.m" -type f -exec grep -l -E "(char|Char|CHAR)" {} \; | wc -l) -gt 10 ]; then
  echo "... and $(expr $(find "$SEARCH_DIR" -name "*.m" -type f -exec grep -l -E "(char|Char|CHAR)" {} \; | wc -l) - 10) more files"
fi