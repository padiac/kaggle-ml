#!/bin/bash

# scripts/kaggle_download.sh
# Usage: ./scripts/kaggle_download.sh <competition-slug>

COMPETITION=$1

if [ -z "$COMPETITION" ]; then
    echo "Usage: $0 <competition-slug>"
    exit 1
fi

DATA_DIR="data/$COMPETITION"

echo "Downloading data for $COMPETITION into $DATA_DIR..."

mkdir -p "$DATA_DIR"

kaggle competitions download -c "$COMPETITION" -p "$DATA_DIR"

if [ $? -eq 0 ]; then
    echo "Download successful based on exit code."
else
    echo "Download failed. Please check your Kaggle API credentials and competition name."
    exit 1
fi

echo "Unzipping..."
unzip -o "$DATA_DIR/$COMPETITION.zip" -d "$DATA_DIR"
rm "$DATA_DIR/$COMPETITION.zip"

echo "Done. Data is in $DATA_DIR"
