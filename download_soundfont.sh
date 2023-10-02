#!/bin/bash

# Check if the target directory is specified
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <target_directory>"
    exit 1
fi

# Set the target directory
TARGET_DIR="$1"

# Check if the target directory exists. If not, create it.
if [ ! -d "$TARGET_DIR" ]; then
    echo "Target directory $TARGET_DIR does not exist. Creating..."
    mkdir -p "$TARGET_DIR"
    if [ $? -ne 0 ]; then
        echo "Error creating target directory $TARGET_DIR"
        exit 1
    fi
fi

# Check https://packages.ubuntu.com/jammy/all/fluid-soundfont-gm/download for a
# complete list of mirrors
URL="http://mirrors.kernel.org/ubuntu/pool/universe/f/fluid-soundfont/fluid-soundfont-gm_3.1-5.3_all.deb"

# Download the deb package
echo "Downloading the deb package..."
wget -q $URL -O fluid-soundfont-gm.deb

if [ $? -ne 0 ]; then
    echo "Error downloading the deb package"
    exit 1
fi

# Create a temporary directory
TMP_DIR=$(mktemp -d)

# Extract the data.tar.zst component from the deb package using ar
echo "Extracting the data.tar.zst from the deb package..."
ar -p fluid-soundfont-gm.deb data.tar.zst > $TMP_DIR/data.tar.zst

if [ $? -ne 0 ]; then
    echo "Error extracting the data.tar.zst component from the deb package"
    rm -rf $TMP_DIR
    exit 1
fi

echo "Extracting the contents of data.tar.zst..."
unzstd -o $TMP_DIR/data.tar $TMP_DIR/data.tar.zst
tar -xf $TMP_DIR/data.tar -C $TMP_DIR

if [ $? -ne 0 ]; then
    echo "Error extracting the contents of data.tar.zst"
    rm -rf $TMP_DIR
    exit 1
fi

# Find the .sf2 file and move it to the target directory
echo "Moving the .sf2 file to $TARGET_DIR"
find $TMP_DIR -type f -name '*.sf2' -exec mv {} $TARGET_DIR \;

# Clean up
rm -rf $TMP_DIR
rm fluid-soundfont-gm.deb

echo "Done."

