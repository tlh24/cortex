#!/usr/bin/bash
mkdir -p /tmp/ec3
mkdir -p /tmp/ec3/verify_database
./_build/default/verify.exe "$@"
