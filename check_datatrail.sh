#!/bin/bash

events=()
failed=()

for ev in "${events[@]}"; do
    if ! datatrail ls chime.event.intensity.raw "$ev"; then
        echo "Error processing event $ev" >&2
        failed+=("$ev")
    fi
done

if (( ${#failed[@]} > 0 )); then
    echo "There were errors in ${#failed[@]} event(s): ${failed[*]}" >&2
    exit 1
else
    echo "All events processed successfully."
fi