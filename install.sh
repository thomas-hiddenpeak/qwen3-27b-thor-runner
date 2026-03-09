#!/bin/bash
# Qwen3.5-Thor — Quick Install Script
# Downloads the pre-built binary for Jetson AGX Thor (aarch64, SM110a)
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/thomas-hiddenpeak/qwen35-thor/master/install.sh | bash
#   # or
#   wget -qO- https://raw.githubusercontent.com/thomas-hiddenpeak/qwen35-thor/master/install.sh | bash

set -euo pipefail

REPO="thomas-hiddenpeak/qwen35-thor"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
BINARY_NAME="qwen35-thor"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

echo ""
echo -e "${CYAN}  Qwen3.5-Thor Inference Engine — Installer${NC}"
echo -e "  NVIDIA Jetson AGX Thor • SM110a Blackwell • 128GB LPDDR5X"
echo ""

# --- Check platform ---
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    error "This binary is built for aarch64 (Jetson AGX Thor). Detected: $ARCH"
fi

# --- Check for download tool ---
if command -v curl &>/dev/null; then
    FETCH="curl -fSL"
    FETCH_QUIET="curl -fsSL"
elif command -v wget &>/dev/null; then
    FETCH="wget -O-"
    FETCH_QUIET="wget -qO-"
else
    error "Neither curl nor wget found. Please install one first."
fi

# --- Get latest release tag ---
info "Fetching latest release..."
if command -v curl &>/dev/null; then
    LATEST_TAG=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')
else
    LATEST_TAG=$(wget -qO- "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')
fi

if [ -z "$LATEST_TAG" ]; then
    error "Failed to fetch latest release tag from GitHub."
fi

VERSION="${LATEST_TAG#v}"
ASSET_NAME="${BINARY_NAME}-${LATEST_TAG}-aarch64-sm110a"
DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${LATEST_TAG}/${ASSET_NAME}"

info "Latest release: ${LATEST_TAG}"
info "Downloading: ${ASSET_NAME}"

# --- Create install directory ---
mkdir -p "$INSTALL_DIR"

# --- Download binary ---
DEST="${INSTALL_DIR}/${BINARY_NAME}"
if ! $FETCH "$DOWNLOAD_URL" -o "$DEST" 2>/dev/null; then
    # wget uses different syntax for output
    $FETCH "$DOWNLOAD_URL" > "$DEST" 2>/dev/null || error "Download failed: $DOWNLOAD_URL"
fi

chmod +x "$DEST"
ok "Installed: ${DEST}"

# --- Verify ---
if "$DEST" version 2>/dev/null; then
    echo ""
else
    warn "Binary downloaded but failed to run. Check CUDA/driver compatibility."
fi

# --- Check PATH ---
if ! echo "$PATH" | tr ':' '\n' | grep -qx "$INSTALL_DIR"; then
    echo ""
    warn "$INSTALL_DIR is not in your PATH."
    echo "  Add it to your shell profile:"
    echo ""
    echo "    echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> ~/.bashrc"
    echo "    source ~/.bashrc"
    echo ""
fi

# --- Download config templates ---
info "Downloading config templates..."
CONFIGS_DIR="${INSTALL_DIR}/../share/qwen35-thor/configs"
mkdir -p "$CONFIGS_DIR"

CONFIG_FILES="qwen3.5-4b.conf qwen3.5-9b.conf qwen3.5-27b.conf qwen3.5-27b-nvfp4.conf qwen3.5-35b-a3b.conf qwen3.5-122b-a10b-nvfp4.conf serve.conf engine.conf"
CONFIGS_BASE="https://raw.githubusercontent.com/${REPO}/${LATEST_TAG}/configs"

for cf in $CONFIG_FILES; do
    if $FETCH_QUIET "${CONFIGS_BASE}/${cf}" > "${CONFIGS_DIR}/${cf}" 2>/dev/null; then
        true
    else
        warn "Failed to download ${cf}"
    fi
done
ok "Configs saved to: ${CONFIGS_DIR}"

# --- Usage hints ---
echo ""
echo -e "${GREEN}=== Installation Complete ===${NC}"
echo ""
echo "  Quick start:"
echo ""
echo "    # 1. Edit config — set model_dir to your Qwen3.5 weights path"
echo "    nano ${CONFIGS_DIR}/qwen3.5-27b.conf"
echo ""
echo "    # 2. Start API server"
echo "    ${BINARY_NAME} serve --config ${CONFIGS_DIR}/qwen3.5-27b.conf"
echo ""
echo "    # 3. Or specify model directory directly"
echo "    ${BINARY_NAME} serve --model-dir /path/to/Qwen3.5-27B --kv-cache-gb 8"
echo ""
echo "    # 4. Interactive chat"
echo "    ${BINARY_NAME} chat --config ${CONFIGS_DIR}/qwen3.5-9b.conf"
echo ""
echo "  Supported models:"
echo "    Qwen3.5-4B / 9B / 27B (BF16)"
echo "    Qwen3.5-27B-NVFP4 (W4A16)"
echo "    Qwen3.5-35B-A3B (MoE BF16)"
echo "    Qwen3.5-122B-A10B-NVFP4 (MoE FP4)"
echo ""
echo "  Model weights: https://huggingface.co/Qwen"
echo ""
