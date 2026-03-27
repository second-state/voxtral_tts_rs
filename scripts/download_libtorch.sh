#!/bin/bash
# Download libtorch from second-state/libtorch-releases (no Python required)
# Usage: bash scripts/download_libtorch.sh [cpu|cuda]
#
# Auto-detects architecture (x86_64 or aarch64).

set -e

VARIANT="${1:-cpu}"
LIBTORCH_VERSION="2.7.1"
BASE_URL="https://github.com/second-state/libtorch-releases/releases/download/v${LIBTORCH_VERSION}"

# Detect architecture
case "$(uname -m)" in
    x86_64|amd64)  ARCH="x86_64" ;;
    aarch64|arm64) ARCH="aarch64" ;;
    *)
        echo "Unsupported architecture: $(uname -m)"
        exit 1
        ;;
esac

case "${VARIANT}" in
    cpu)
        URL="${BASE_URL}/libtorch-cxx11-abi-${ARCH}-${LIBTORCH_VERSION}.tar.gz"
        ;;
    cuda)
        URL="${BASE_URL}/libtorch-cxx11-abi-${ARCH}-cuda12.6-${LIBTORCH_VERSION}.tar.gz"
        ;;
    *)
        echo "Usage: $0 [cpu|cuda]"
        echo "  cpu  - CPU only (default)"
        echo "  cuda - CUDA 12.6"
        echo ""
        echo "Architecture is auto-detected (current: ${ARCH})."
        exit 1
        ;;
esac

echo "Downloading libtorch ${LIBTORCH_VERSION} (${VARIANT}, ${ARCH})..."
curl -fSL -o libtorch.tar.gz "${URL}"

echo "Extracting..."
tar xzf libtorch.tar.gz
rm libtorch.tar.gz

echo ""
echo "libtorch downloaded to: $(pwd)/libtorch"
echo ""
echo "Set the environment variables before building:"
echo "  export LIBTORCH=$(pwd)/libtorch"
echo "  export LIBTORCH_BYPASS_VERSION_CHECK=1"
echo ""
echo "Then build with:"
echo "  cargo build --release"
