#!/usr/bin/env bash
#
# Build sd.cpp + ggml + our C wrapper into a fat XCFramework that covers:
#   - macos-arm64      (Apple Silicon Macs)
#   - ios-arm64        (iPhones / iPads)
#   - ios-arm64_x86_64-simulator (Xcode iOS Simulator)
#
# Output: Frameworks/sdcpp.xcframework
#
# Usage:
#   ./Scripts/build-xcframework.sh            # all slices
#   ./Scripts/build-xcframework.sh macos      # macOS arm64 only (fastest, for tests)
#   ./Scripts/build-xcframework.sh ios        # iOS device + simulator
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SD_DIR="$ROOT/Sources/CMirage/vendor/sd-cpp"
WRAP_DIR="$ROOT/Sources/CMirage"
OUT_DIR="$ROOT/Frameworks"
BUILD_ROOT="$ROOT/.build/xcframework"

mkdir -p "$OUT_DIR" "$BUILD_ROOT"

WHICH="${1:-all}"

# ----- helpers -----

build_slice() {
    local PLATFORM=$1    # MAC_ARM64 | IOS | SIMULATOR_ARM64
    local SYSROOT=$2     # path or "macos"
    local CMAKE_OSX_DEPLOYMENT_TARGET=$3
    local ARCHS=$4
    local NAME=$5        # output dir suffix
    local EXTRA_CMAKE="${6:-}"

    local BUILD_DIR="$BUILD_ROOT/$NAME"
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"

    echo "==> Configuring sd.cpp for $NAME"
    if [ "$SYSROOT" = "macos" ]; then
        cmake -S "$SD_DIR" -B "$BUILD_DIR" \
            -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DSD_METAL=ON \
            -DSD_BUILD_EXAMPLES=OFF \
            -DSD_BUILD_SERVER=OFF \
            -DSD_BUILD_SHARED_LIBS=OFF \
            -DCMAKE_OSX_ARCHITECTURES="$ARCHS" \
            -DCMAKE_OSX_DEPLOYMENT_TARGET="$CMAKE_OSX_DEPLOYMENT_TARGET" \
            $EXTRA_CMAKE
    else
        cmake -S "$SD_DIR" -B "$BUILD_DIR" \
            -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DSD_METAL=ON \
            -DSD_BUILD_EXAMPLES=OFF \
            -DSD_BUILD_SERVER=OFF \
            -DSD_BUILD_SHARED_LIBS=OFF \
            -DCMAKE_SYSTEM_NAME=iOS \
            -DCMAKE_OSX_SYSROOT="$SYSROOT" \
            -DCMAKE_OSX_ARCHITECTURES="$ARCHS" \
            -DCMAKE_OSX_DEPLOYMENT_TARGET="$CMAKE_OSX_DEPLOYMENT_TARGET" \
            -DCMAKE_C_FLAGS="-fembed-bitcode" \
            -DCMAKE_CXX_FLAGS="-fembed-bitcode" \
            $EXTRA_CMAKE
    fi

    echo "==> Building sd.cpp for $NAME"
    ninja -C "$BUILD_DIR" stable-diffusion

    echo "==> Compiling our C wrapper for $NAME"
    local WRAP_OBJ="$BUILD_DIR/MirageC.o"
    local SDK_FLAG=""
    local TARGET_FLAG=""
    case "$NAME" in
        macos-arm64)
            SDK_FLAG=""
            TARGET_FLAG="-target arm64-apple-macos$CMAKE_OSX_DEPLOYMENT_TARGET"
            ;;
        ios-arm64)
            SDK_FLAG="-isysroot $(xcrun -sdk iphoneos --show-sdk-path)"
            TARGET_FLAG="-target arm64-apple-ios$CMAKE_OSX_DEPLOYMENT_TARGET"
            ;;
        ios-sim)
            SDK_FLAG="-isysroot $(xcrun -sdk iphonesimulator --show-sdk-path)"
            TARGET_FLAG="-target arm64-apple-ios$CMAKE_OSX_DEPLOYMENT_TARGET-simulator"
            ;;
    esac
    clang++ -c -std=c++17 -O2 \
        $SDK_FLAG $TARGET_FLAG \
        -I"$WRAP_DIR/include" \
        -I"$SD_DIR/include" \
        -I"$SD_DIR/ggml/include" \
        "$WRAP_DIR/sd/MirageC.cpp" \
        -o "$WRAP_OBJ"

    # Combine all the static libs into one fat archive so the binary target
    # has a single .a to link. libwebp / libwebm are CLI-only deps that
    # only get built when sd.cpp's `examples` target is requested — we
    # build only the library, so skip them.
    local OUT_LIB="$BUILD_DIR/libmirage-sdcpp.a"
    rm -f "$OUT_LIB"
    libtool -static -o "$OUT_LIB" \
        "$BUILD_DIR/libstable-diffusion.a" \
        "$BUILD_DIR/ggml/src/libggml.a" \
        "$BUILD_DIR/ggml/src/libggml-base.a" \
        "$BUILD_DIR/ggml/src/libggml-cpu.a" \
        "$BUILD_DIR/ggml/src/ggml-metal/libggml-metal.a" \
        "$BUILD_DIR/ggml/src/ggml-blas/libggml-blas.a" \
        "$WRAP_OBJ"
    echo "==> $OUT_LIB ($(du -h "$OUT_LIB" | cut -f1))"
}

case "$WHICH" in
    macos|all)
        build_slice MAC_ARM64 macos "14.0" "arm64" "macos-arm64"
        ;;
esac
case "$WHICH" in
    ios|all)
        build_slice IOS "$(xcrun -sdk iphoneos --show-sdk-path)" "17.0" "arm64" "ios-arm64"
        build_slice SIMULATOR_ARM64 "$(xcrun -sdk iphonesimulator --show-sdk-path)" "17.0" "arm64" "ios-sim"
        ;;
esac

# ----- assemble XCFramework -----

echo "==> Assembling sdcpp.xcframework"
rm -rf "$OUT_DIR/sdcpp.xcframework"

XC_ARGS=""
for slice in macos-arm64 ios-arm64 ios-sim; do
    LIB="$BUILD_ROOT/$slice/libmirage-sdcpp.a"
    if [ -f "$LIB" ]; then
        XC_ARGS="$XC_ARGS -library $LIB -headers $WRAP_DIR/include"
    fi
done

if [ -z "$XC_ARGS" ]; then
    echo "No slices built — nothing to assemble." >&2
    exit 1
fi

xcodebuild -create-xcframework $XC_ARGS -output "$OUT_DIR/sdcpp.xcframework"

echo
echo "==> Done: $OUT_DIR/sdcpp.xcframework"
ls -lh "$OUT_DIR/sdcpp.xcframework"
