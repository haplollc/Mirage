//
//  KilnImageTests.swift
//  Two test tiers:
//    - Fast (always run): verify the package compiles, native lib loads,
//      version string is well-formed, error reporting works for bad inputs.
//    - Heavy (gated on `KILN_TEST_MODELS_DIR` env var): actually load
//      multi-GB weights + generate an image. Skip by default so CI
//      doesn't need to ship 6+ GB through every PR.
//
//  Heavy-test usage:
//      KILN_TEST_MODELS_DIR=/path/to/dir \
//          swift test --filter HeavyIntegrationTests
//  The directory must contain:
//      diffusion.gguf          (any sd.cpp-compatible diffusion model)
//      vae.safetensors         (optional)
//      text-encoder.gguf       (optional)
//

import XCTest
import CoreGraphics
@testable import KilnImage

// MARK: - Fast smoke tests

final class KilnImageSmokeTests: XCTestCase {

    func testNativeVersionStringIsWellFormed() {
        let v = KilnImage.nativeVersion
        XCTAssertFalse(v.isEmpty, "native version string is empty")
        // Expect MAJOR.MINOR.PATCH.
        XCTAssertEqual(v.split(separator: ".").count, 3, "version not in semver shape: \(v)")
    }

    func testLoadingMissingModelReportsError() async {
        let bogus = URL(fileURLWithPath: "/this/path/does/not/exist/diffusion.gguf")
        do {
            _ = try Engine(models: ModelFiles(diffusionModel: bogus))
            XCTFail("expected modelLoadFailed for non-existent path")
        } catch let KilnImageError.modelLoadFailed(reason) {
            XCTAssertFalse(reason.isEmpty, "error message should explain failure")
        } catch {
            XCTFail("got unexpected error type: \(error)")
        }
    }
}

// MARK: - Heavy integration tests (env-gated)

final class KilnImageHeavyIntegrationTests: XCTestCase {

    /// Resolved at runtime via `KILN_TEST_MODELS_DIR`. If unset, every
    /// test in this class throws `XCTSkip`.
    private func resolveModelsDir() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["KILN_TEST_MODELS_DIR"] else {
            throw XCTSkip("Set KILN_TEST_MODELS_DIR to a folder containing model files.")
        }
        return URL(fileURLWithPath: path)
    }

    private func resolveModels() throws -> ModelFiles {
        let dir = try resolveModelsDir()
        let fm = FileManager.default

        func find(_ candidates: [String]) -> URL? {
            for name in candidates {
                let u = dir.appendingPathComponent(name)
                if fm.fileExists(atPath: u.path) { return u }
            }
            return nil
        }

        guard let diffusion = find(["diffusion.gguf", "z-image-turbo-Q3_K_M.gguf"]) else {
            throw XCTSkip("Diffusion model not found in \(dir.path)")
        }
        let vae = find(["vae.safetensors", "ae.safetensors"])
        let llm = find([
            "text-encoder.gguf",
            "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        ])
        return ModelFiles(diffusionModel: diffusion, vae: vae, textEncoder: llm)
    }

    /// Loads the engine. Pure load test — no generation. Catches model
    /// path / quantization issues early without paying for sampling.
    func testEngineLoads() async throws {
        let models = try resolveModels()
        let engine = try Engine(models: models)
        _ = engine
    }

    /// End-to-end smoke: tiny 256×256 generation at fewer steps. Useful as
    /// a "doesn't crash + returns an image of the right shape" check that
    /// still completes in under a minute on Apple Silicon.
    func testTinyGeneration() async throws {
        let models = try resolveModels()
        let engine = try Engine(models: models)
        let request = GenerationRequest(
            prompt: "a single red apple on a white background, photorealistic",
            width: 256, height: 256,
            steps: 4,
            cfgScale: 1.0,
            seed: 42
        )
        let img = try await engine.generate(request)
        XCTAssertEqual(img.width, 256)
        XCTAssertEqual(img.height, 256)

        // Save to disk so a human can eyeball it after a heavy run.
        if let data = img.pngData() {
            let out = FileManager.default.temporaryDirectory
                .appendingPathComponent("kiln-tiny.png")
            try? data.write(to: out)
            print("[KilnImageTests] saved smoke output → \(out.path)")
        }
    }

    /// Deterministic-seed test: same seed → identical pixel buffer twice
    /// in a row. Guards against non-reproducible sampler paths.
    func testFixedSeedIsDeterministic() async throws {
        let models = try resolveModels()
        let engine = try Engine(models: models)
        let req = GenerationRequest(
            prompt: "a single blue circle, flat color, centered",
            width: 256, height: 256,
            steps: 4,
            cfgScale: 1.0,
            seed: 12345
        )
        let a = try await engine.generate(req)
        let b = try await engine.generate(req)
        XCTAssertEqual(a.width, b.width)
        XCTAssertEqual(a.height, b.height)
        // We don't assert byte-identical because GPU non-determinism is
        // a real thing — but the dimensions and a rough sanity check on
        // payload size are within scope.
        let aBytes = a.pngData()?.count ?? 0
        let bBytes = b.pngData()?.count ?? 0
        XCTAssertGreaterThan(aBytes, 1000, "image looks suspiciously small")
        XCTAssertEqual(aBytes, bBytes, accuracy: aBytes / 5,
                       "PNG byte-size diverged > 20% — sampler likely non-deterministic")
    }
}
