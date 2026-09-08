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
    # GGML_METAL_EMBED_LIBRARY embeds the .metal source as a binary blob in
    # libggml-metal.a so the runtime compiles it via newLibraryWithSource
    # instead of newLibraryWithURL. Without it, ggml-metal_library_init scans
    # the app bundle for a `default.metallib` resource — and on iOS we ship
    # alongside Kuzco's llama.cpp `llama_llama.bundle/default.metallib`, which
    # has a stale ggml-metal kernel set missing bf16 mul_mm + the new fused
    # norm bindings sd.cpp expects. Embedding eliminates the collision.
    if [ "$SYSROOT" = "macos" ]; then
        cmake -S "$SD_DIR" -B "$BUILD_DIR" \
            -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DSD_METAL=ON \
            -DGGML_METAL=ON \
            -DGGML_METAL_EMBED_LIBRARY=ON \
            -DSD_BUILD_EXAMPLES=OFF \
            -DSD_BUILD_SERVER=OFF \
            -DSD_BUILD_SHARED_LIBS=OFF \
            -DSD_WEBP=OFF \
            -DSD_WEBM=OFF \
            -DCMAKE_OSX_ARCHITECTURES="$ARCHS" \
            -DCMAKE_OSX_DEPLOYMENT_TARGET="$CMAKE_OSX_DEPLOYMENT_TARGET" \
            $EXTRA_CMAKE
    else
        cmake -S "$SD_DIR" -B "$BUILD_DIR" \
            -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DSD_METAL=ON \
            -DGGML_METAL=ON \
            -DGGML_METAL_EMBED_LIBRARY=ON \
            -DSD_BUILD_EXAMPLES=OFF \
            -DSD_BUILD_SERVER=OFF \
            -DSD_BUILD_SHARED_LIBS=OFF \
            -DSD_WEBP=OFF \
            -DSD_WEBM=OFF \
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

    # Partial-link all sd.cpp/ggml objects into a single .o, hiding every
    # symbol except the public mirage_* C ABI. This is essential: HaploAI
    # also links Kuzco's vendored llama.cpp, which defines its own (older)
    # `_ggml_metal_library_init` etc. Without symbol hiding, Apple's ld
    # picks the first match (usually Kuzco's) and sd.cpp ends up calling
    # the wrong, stale, bundle-metallib-loading ggml-metal at runtime,
    # crashing on missing kernels (kernel_mul_mm_bf16_f32, fused norm
    # binding mismatches, etc.).
    #
    # `ld -r` + `-exported_symbols_list` keeps mirage_* exported as `T`
    # and demotes everything else to local (`t`), so the app linker
    # never sees Mirage's ggml as a candidate for Kuzco's references.
    local PRELINK_OBJ="$BUILD_DIR/libmirage-sdcpp-prelinked.o"
    local SYMS_LIST="$ROOT/Scripts/mirage-exported-symbols.txt"
    local PLATFORM_VERSION_FLAG=""
    case "$NAME" in
        macos-arm64)  PLATFORM_VERSION_FLAG="-platform_version macos $CMAKE_OSX_DEPLOYMENT_TARGET $CMAKE_OSX_DEPLOYMENT_TARGET" ;;
        ios-arm64)    PLATFORM_VERSION_FLAG="-platform_version ios $CMAKE_OSX_DEPLOYMENT_TARGET $CMAKE_OSX_DEPLOYMENT_TARGET" ;;
        ios-sim)      PLATFORM_VERSION_FLAG="-platform_version ios-simulator $CMAKE_OSX_DEPLOYMENT_TARGET $CMAKE_OSX_DEPLOYMENT_TARGET" ;;
    esac

    rm -f "$PRELINK_OBJ"
    ld -r -arch arm64 $PLATFORM_VERSION_FLAG \
        -exported_symbols_list "$SYMS_LIST" \
        -force_load "$BUILD_DIR/libstable-diffusion.a" \
        -force_load "$BUILD_DIR/ggml/src/libggml.a" \
        -force_load "$BUILD_DIR/ggml/src/libggml-base.a" \
        -force_load "$BUILD_DIR/ggml/src/libggml-cpu.a" \
        -force_load "$BUILD_DIR/ggml/src/ggml-metal/libggml-metal.a" \
        -force_load "$BUILD_DIR/ggml/src/ggml-blas/libggml-blas.a" \
        "$WRAP_OBJ" \
        -o "$PRELINK_OBJ"

    # Sanity-check: only the six mirage_* symbols should be globally exported.
    # NB: use `set +e` around grep — when symbol hiding works, the grep finds
    # nothing and returns non-zero, which would otherwise trip pipefail and
    # silently kill the script.
    local EXPORTED_COUNT OTHER_GGML
    set +e
    EXPORTED_COUNT=$(nm -gU "$PRELINK_OBJ" 2>/dev/null | grep -cE " T _mirage_")
    OTHER_GGML=$(nm -gU "$PRELINK_OBJ" 2>/dev/null | grep -cE " T _ggml_metal_library_init$")
    set -e
    echo "==> [$NAME] exported mirage_* symbols = $EXPORTED_COUNT, leaked ggml_metal_library_init = $OTHER_GGML"
    if [ "$OTHER_GGML" != "0" ]; then
        echo "    ERROR: ggml_metal_library_init is still globally exported — symbol hiding failed" >&2
        exit 1
    fi
    if [ "$EXPORTED_COUNT" = "0" ]; then
        echo "    ERROR: no mirage_* symbols exported — exports list misapplied" >&2
        exit 1
    fi

    # Wrap the relinked .o in a static archive so SPM's binary target accepts it.
    local OUT_LIB="$BUILD_DIR/libmirage-sdcpp.a"
    rm -f "$OUT_LIB"
    libtool -static -o "$OUT_LIB" "$PRELINK_OBJ"
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
