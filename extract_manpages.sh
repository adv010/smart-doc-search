#!/bin/bash

# Check if argument is provided
if [ $# -eq 0 ]; then
  echo "Error: Please specify the linux command to retrieve the manpage (USAGE: './man_to_txt.sh ls - retrieves txtfile of 'ls' manpage')"
  exit 1
fi

OUTPUT_DIR="./manpages"
mkdir -p "$OUTPUT_DIR"

COMMAND="$1"
OUTPUT_FILE="${OUTPUT_DIR}/${COMMAND}.txt"  # <-- FIXED: Added missing "

# Convert man page to clean plaintext
man "$COMMAND" | groff -Tascii -man | col -b > "$OUTPUT_FILE"

echo "Man page for '$COMMAND' saved to: $OUTPUT_FILE"