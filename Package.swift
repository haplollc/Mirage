// swift-tools-version:5.9
import PackageDescription

// MARK: - Mirage SPM package
//
// The native engine (sd.cpp + ggml + Metal backend + our C wrapper) ships
// as a prebuilt XCFramework downloaded from this tag's GitHub release.
// SPM consumers fetching `.package(url:, from: "0.2.0")` get the binary
// transparently — no cmake / ninja / iOS SDK tooling required locally.

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
        // Pre-built sd.cpp + ggml + Metal + our C wrapper. The xcframework
        // already ships the `CMirage` modulemap + headers internally, so
        // Swift can `import CMirage` by depending directly on this binary
        // target. A separate Swift target re-exporting the same headers
        // would conflict with the xcframework's modulemap when integrated
        // into Xcode projects.
        .binaryTarget(
            name: "sdcpp",
            url: "https://github.com/haplollc/Mirage/releases/download/0.2.0/sdcpp.xcframework.zip",
            checksum: "2754f948ff12e1c546f46e0dfa59f29627d468b1c30ba1783bdaa5d8948e5472"
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
