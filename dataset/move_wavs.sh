#!/bin/bash

# Define the source (big folder) and destination (small folder)
SOURCE_FOLDER="/workspace/AcademiCodec/dataset/LibriTTS/test-clean"
DEST_FOLDER="/workspace/AcademiCodec/dataset/LibriTTS/test-clean-2"
LIST_FILE="/workspace/AcademiCodec/dataset/LibriTTS/test-clean-2.lst"

# Create the destination folder if it doesn't exist
mkdir -p "$DEST_FOLDER"

# Read each line from the .lst file and move the file
while IFS= read -r file; do
    # Ensure the file exists before moving
    if [ -f "$SOURCE_FOLDER/$file" ]; then
        mv "$SOURCE_FOLDER/$file" "$DEST_FOLDER/"
        echo "Moved: $file"
    else
        echo "File not found: $file"
    fi
done < "$LIST_FILE"

echo "✅ All files from $LIST_FILE moved to $DEST_FOLDER."