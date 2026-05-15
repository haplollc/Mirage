// swift-tools-version:5.9
import PackageDescription

// MARK: - Mirage SPM package
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
    name: "Mirage",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
        .visionOS(.v1),
        .macCatalyst(.v17),
    ],
    products: [
        .library(name: "Mirage", targets: ["Mirage"]),
    ],
    targets: [
        // Pre-built sd.cpp + ggml + Metal + our C wrapper. Produced by
        // `Scripts/build-xcframework.sh`. The xcframework already ships
        // the `CMirage` modulemap + headers internally, so Swift can
        // `import CMirage` by depending directly on this binary target.
        // A separate Swift target re-exporting the same headers would
        // conflict with the xcframework's modulemap when integrated into
        // Xcode projects.
        .binaryTarget(
            name: "sdcpp",
            path: "Frameworks/sdcpp.xcframework"
        ),
        // Public Swift API. Imports CMirage (provided by the xcframework).
        .target(
            name: "Mirage",
            dependencies: ["sdcpp"],
            path: "Sources/Mirage"
        ),
        // Unit tests. The integration test that loads multi-GB weights is
        // gated on env var MIRAGE_TEST_MODELS_DIR — keep CI fast by default.
        .testTarget(
            name: "MirageTests",
            dependencies: ["Mirage"],
            path: "Tests/MirageTests"
        ),
    ],
    cxxLanguageStandard: .cxx17
)
