#!/bin/sh
# Explicit opt-in only. No inference or test invokes this script.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
dest=${1:-"$root/models/qwen"}
lock="$root/spec/references.lock.json"
revision=$(jq -r .model.revision "$lock")
mkdir -p "$dest"
for name in config.json tokenizer.json tokenizer_config.json chat_template.jinja model.safetensors.index.json model.safetensors-00001-of-00001.safetensors; do
    expected=$(jq -r --arg n "$name" '.model.files[] | select(.rfilename==$n) | .lfs.sha256 // empty' "$lock")
    blob=$(jq -r --arg n "$name" '.model.files[] | select(.rfilename==$n) | .blobId' "$lock")
    valid=false
    if [ -f "$dest/$name" ]; then
        if [ -n "$expected" ]; then actual=$(sha256sum "$dest/$name" | cut -d ' ' -f 1); else actual=$(git hash-object "$dest/$name"); expected=$blob; fi
        [ "$actual" != "$expected" ] || valid=true
    fi
    if [ "$valid" = false ]; then
        curl --fail --location --retry 3 --output "$dest/$name.part" "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/$revision/$name"
        if [ -n "$expected" ] && [ "$expected" != "$blob" ]; then actual=$(sha256sum "$dest/$name.part" | cut -d ' ' -f 1); else actual=$(git hash-object "$dest/$name.part"); expected=$blob; fi
        [ "$actual" = "$expected" ] || { echo "hash mismatch: $name" >&2; exit 1; }
        mv "$dest/$name.part" "$dest/$name"
    fi
    echo "verified: $name"
done
printf '%s\n' "$revision" > "$dest/REVISION"
