#!/bin/bash

archive_production_config() {
    local config_path=$1
    local config_store=$2
    local array_id=${3:-0}
    local array_min=${4:-0}
    local config_dest

    mkdir -p "${config_store}"
    config_dest="${config_store}/$(basename "${config_path}")"

    if [[ -e "${config_dest}" ]]; then
        if cmp -s "${config_path}" "${config_dest}"; then
            return 0
        fi
        echo "stored config mismatch: ${config_dest} differs from ${config_path}" >&2
        return 1
    fi

    if [[ "${array_id}" != "${array_min}" ]]; then
        return 0
    fi

    cp "${config_path}" "${config_dest}"
    if ! cmp -s "${config_path}" "${config_dest}"; then
        echo "stored config copy verification failed: ${config_dest}" >&2
        return 1
    fi
}
