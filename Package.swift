// swift-tools-version:5.9
import PackageDescription

// MARK: - KilnImage SPM package
//
// The native engine (sd.cpp + ggml + Metal backend + our C wrapper) ships
// as a prebuilt XCFramework (`Frameworks/sdcpp.xcframework`) so SPM
// consumers don't need cmake / ninja / iOS SDK tooling installed.
//
// The XCFramework is produced by `Scripts/build-xcframework.sh`. Run it
// after a fresh checkout, then `swift build` / `swift test` work normally.
// The XCFramework is also shipped as an asset on each tagged GitHub
// release for SPM consumers that fetch this package by URL.

let package = Package(
    name: "KilnImage",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
        .visionOS(.v1),
        .macCatalyst(.v17),
    ],
    products: [
        .library(name: "KilnImage", targets: ["KilnImage"]),
    ],
    targets: [
        // Pre-built sd.cpp + ggml + Metal + our C wrapper. Produced by
        // `Scripts/build-xcframework.sh`.
        .binaryTarget(
            name: "sdcpp",
            path: "Frameworks/sdcpp.xcframework"
        ),
        // C bridge header that Swift imports. The actual `kiln_*` symbols
        // live inside the `sdcpp` binary target; this target only contains
        // the header + module map so Swift can name them.
        .target(
            name: "CKilnImage",
            dependencies: ["sdcpp"],
            path: "Sources/CKilnImage",
            exclude: ["vendor", "sd"],
            sources: [],
            publicHeadersPath: "include"
        ),
        // Public Swift API.
        .target(
            name: "KilnImage",
            dependencies: ["CKilnImage"],
            path: "Sources/KilnImage"
        ),
        // Unit tests. The integration test that loads multi-GB weights is
        // gated on env var KILN_TEST_MODELS_DIR — keep CI fast by default.
        .testTarget(
            name: "KilnImageTests",
            dependencies: ["KilnImage"],
            path: "Tests/KilnImageTests"
        ),
    ],
    cxxLanguageStandard: .cxx17
)
