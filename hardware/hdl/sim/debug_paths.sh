#!/bin/bash

# Simple test script to debug paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
HDL_DIR="$SCRIPT_DIR/.."
RTL_DIR="$HDL_DIR/rtl"
TB_DIR="$HDL_DIR/tb" 

echo "SCRIPT_DIR = $SCRIPT_DIR"
echo "HDL_DIR = $HDL_DIR"
echo "RTL_DIR = $RTL_DIR"
echo "TB_DIR = $TB_DIR"
echo ""
echo "TB file exists: $(test -f "$TB_DIR/tb_top.v" && echo "YES" || echo "NO")"
echo "RTL directory exists: $(test -d "$RTL_DIR" && echo "YES" || echo "NO")"

# Test finding sources
echo ""
echo "RTL sources found:"
find "$RTL_DIR" -name "*.v" -o -name "*.sv" | head -5

echo ""
echo "TB file path: $TB_DIR/tb_top.v"
ls -la "$TB_DIR/tb_top.v"
