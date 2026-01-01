#!/bin/bash

# scripts/new_competition.sh
# Usage: ./scripts/new_competition.sh <new-competition-name>

NEW_NAME=$1

if [ -z "$NEW_NAME" ]; then
    echo "Usage: $0 <new-competition-name>"
    exit 1
fi

TARGET_DIR="competitions/$NEW_NAME"
TEMPLATE_SOURCE="competitions/titanic" # We use titanic as the base template

if [ -d "$TARGET_DIR" ]; then
    echo "Directory $TARGET_DIR already exists."
    exit 1
fi

if [ ! -d "$TEMPLATE_SOURCE" ]; then
    echo "Template source $TEMPLATE_SOURCE does NOT exist yet. Please create it first."
    exit 1
fi

echo "Creating new competition: $NEW_NAME"
cp -r "$TEMPLATE_SOURCE" "$TARGET_DIR"

# Clean up any cached files or specific artifacts from the template
rm -f "$TARGET_DIR/submission.csv"
rm -rf "$TARGET_DIR/__pycache__"

# Update the config file with the new name (simple sed replacement)
# We assume the config has `competition_name: titanic`
sed -i "s/competition_name: .*/competition_name: $NEW_NAME/" "$TARGET_DIR/config.yaml"

echo "Created $TARGET_DIR from template."
echo "Don't forget to update $TARGET_DIR/config.yaml with specific columns!"
