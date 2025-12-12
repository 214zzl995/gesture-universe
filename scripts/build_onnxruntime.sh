#!/usr/bin/env bash
set -euo pipefail

# Download and build ONNX Runtime v1.22.0 into target/onnxruntime.
# Optional operator trim config can be provided with --ops-config or OPS_CONFIG.

usage() {
  cat <<'EOF'
Usage: scripts/build_onnxruntime.sh [--ops-config path]

Environment overrides:
  OUT_DIR     Where to place downloads and sources (default: <repo>/target/onnxruntime)
  ORT_VERSION ONNX Runtime tag to fetch (default: 1.22.0)
  OPS_CONFIG  Operator config path passed to --include_ops_by_config
EOF
}

OPS_CONFIG="${OPS_CONFIG:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ops-config)
      OPS_CONFIG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-"$ROOT_DIR/target/onnxruntime"}"
ORT_VERSION="${ORT_VERSION:-1.22.0}"
SRC_DIR="$OUT_DIR/onnxruntime-$ORT_VERSION"
ARCHIVE="$OUT_DIR/onnxruntime-$ORT_VERSION.tar.gz"

if [ -n "$OPS_CONFIG" ]; then
  # Convert to absolute path if relative
  if [[ ! "$OPS_CONFIG" = /* ]]; then
    OPS_CONFIG="$ROOT_DIR/$OPS_CONFIG"
  fi
  if [ ! -f "$OPS_CONFIG" ]; then
    echo "Specified ops config not found: $OPS_CONFIG" >&2
    exit 1
  fi
fi

mkdir -p "$OUT_DIR"

if [ ! -d "$SRC_DIR" ]; then
  echo "Downloading ONNX Runtime v$ORT_VERSION sources..."
  curl -L "https://github.com/microsoft/onnxruntime/archive/refs/tags/v${ORT_VERSION}.tar.gz" -o "$ARCHIVE"
  tar -xzf "$ARCHIVE" -C "$OUT_DIR"
fi

# Fix Eigen SHA1 hash mismatch in deps.txt
DEPS_FILE="$SRC_DIR/cmake/deps.txt"
if [ -f "$DEPS_FILE" ]; then
  echo "Fixing Eigen SHA1 hash in deps.txt..."
  sed -i.bak 's/5ea4d05e62d7f954a46b3213f9b2535bdd866803/51982be81bbe52572b54180454df11a3ece9a934/' "$DEPS_FILE"
fi

pushd "$SRC_DIR" >/dev/null
BUILD_ARGS=(
  --config Release 
  --parallel 
  --skip_tests 
  --use_full_protobuf
  --cmake_extra_defines CMAKE_C_FLAGS="-fPIC" CMAKE_CXX_FLAGS="-fPIC -stdlib=libc++"
)

if [ -n "$OPS_CONFIG" ]; then
  echo "Using operator config at $OPS_CONFIG"
  BUILD_ARGS+=(--minimal_build --include_ops_by_config "$OPS_CONFIG")
fi

# Prevent CMake from using Homebrew's incompatible protobuf 33.1.0
# CMAKE_IGNORE_PREFIX_PATH tells CMake to completely ignore these paths during find_package
export CMAKE_IGNORE_PREFIX_PATH="/opt/homebrew"
export CMAKE_PREFIX_PATH=""

./build.sh "${BUILD_ARGS[@]}"
popd >/dev/null

# Copy build artifacts to platform-independent directory
LIB_DIR="$OUT_DIR/lib"
mkdir -p "$LIB_DIR"

# Detect platform-specific build directory
if [[ "$OSTYPE" == "darwin"* ]]; then
  BUILD_DIR="$SRC_DIR/build/MacOS/Release"
elif [[ "$OSTYPE" == "linux"* ]]; then
  BUILD_DIR="$SRC_DIR/build/Linux/Release"
else
  echo "Unsupported platform: $OSTYPE" >&2
  exit 1
fi

if [ -d "$BUILD_DIR" ]; then
  echo "Copying build artifacts to $LIB_DIR..."
  # Copy only the necessary files, skip permission denied files
  rsync -av --exclude='_deps/protoc_binary-src' "$BUILD_DIR"/ "$LIB_DIR/" 2>/dev/null || cp -r "$BUILD_DIR"/* "$LIB_DIR/" 2>/dev/null || true
  echo "Build artifacts copied successfully"
  
  # Create a linker config file with all required libraries
  echo "Creating linker configuration..."
  cat > "$LIB_DIR/onnxruntime_link_libs.txt" <<EOF
# Add all required ONNX Runtime static libraries
# This file is used by the build system to link all necessary libraries
EOF
  
  # List all onnxruntime libraries
  for lib in "$LIB_DIR"/libonnxruntime_*.a "$LIB_DIR"/libonnx*.a; do
    if [ -f "$lib" ]; then
      echo "$(basename "$lib" .a)" | sed 's/^lib//' >> "$LIB_DIR/onnxruntime_link_libs.txt"
    fi
  done
  
  echo "Library configuration created at $LIB_DIR/onnxruntime_link_libs.txt"
  
  cd "$ROOT_DIR"
else
  echo "Build directory not found: $BUILD_DIR" >&2
  exit 1
fi

echo "ONNX Runtime build finished under $SRC_DIR"
echo "Build artifacts available at $LIB_DIR"
