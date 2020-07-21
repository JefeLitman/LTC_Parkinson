#!/bin/sh

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "usage: <file_id> <file_name>"
    echo "  <file_id> - The file id given by google drive when is shared"
    echo "  <file_name> - The name of the file to be saved"
    exit 1
else
    echo "Downloading $2..."
    internal_command=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
    wget --quiet --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$internal_command&id=$1" -O $2 && rm -rf /tmp/cookies.txt
    echo "Success!"
    exit 0
fi