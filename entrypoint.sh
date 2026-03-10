#!/bin/bash
set -e

HOST_UID="${HOST_UID:-1000}"
HOST_GID="${HOST_GID:-1000}"

# Find or create user with matching UID
EXISTING_USER=$(getent passwd "$HOST_UID" | cut -d: -f1 || true)

if [ -z "$EXISTING_USER" ]; then
    # Create group if GID doesn't exist
    if ! getent group "$HOST_GID" > /dev/null 2>&1; then
        groupadd -g "$HOST_GID" heretic
    fi
    GROUP_NAME=$(getent group "$HOST_GID" | cut -d: -f1)
    useradd -u "$HOST_UID" -g "$GROUP_NAME" -m -s /bin/bash heretic
    EXISTING_USER="heretic"
fi

# Ensure output directories exist and are owned by the target user
for dir in /output /models /workspace; do
    if [ -d "$dir" ]; then
        chown "$HOST_UID:$HOST_GID" "$dir"
    fi
done

# Point HuggingFace cache at the mounted /models volume
export HF_HOME="/models"

# Drop privileges and exec the command as the target user
exec gosu "$EXISTING_USER" "$@"
