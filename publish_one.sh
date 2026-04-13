#!/bin/bash
# Publish a single OxiCUDA crate to crates.io
# Usage: ./publish_one.sh <crate-directory>
# Example: ./publish_one.sh crates/oxicuda-driver

if [ -z "$1" ]; then
    echo "Usage: $0 <crate-directory>"
    echo "Example: $0 crates/oxicuda-driver"
    exit 1
fi

CRATE=$1
echo "===== Publishing $CRATE ====="
cd "$CRATE" || { echo "Directory not found: $CRATE"; exit 1; }
cargo publish --allow-dirty || { echo "Failed to publish: $CRATE"; exit 1; }
echo "Successfully published $CRATE"
