//
//  MirageExampleApp.swift
//
//  Single-file SwiftUI example showing how to wire Mirage into a real
//  app. Tested on iOS 17+ and macOS 14+.
//
//  Setup:
//    1. Put your downloaded model files in the app's documents directory
//       (or anywhere you can construct a URL to). For Z-Image-Turbo:
//
//         documents/
//           z-image-turbo-Q3_K_M.gguf            (3.9 GB)
//           Qwen3-4B-Instruct-2507-Q4_K_M.gguf   (2.3 GB)
//           ae.safetensors                       (320 MB)
//
//       Download bundles from huggingface.co/HaploApps once and copy into
//       the documents dir on first launch.
//
//    2. Drop this file into a fresh Xcode SwiftUI app with `Mirage`
//       added as a Swift package dependency.
//

import SwiftUI
import Mirage

@main
struct MirageExampleApp: App {
    var body: some Scene {
        WindowGroup {
            GenerateView()
        }
    }
}

// MARK: - Generate screen

struct GenerateView: View {
    @State private var prompt: String = "a corgi in a spacesuit on the surface of Mars, photorealistic"
    @State private var image: CGImage?
    @State private var status: String = "Ready."
    @State private var isWorking: Bool = false
    @State private var engine: Engine?

    var body: some View {
        VStack(spacing: 16) {
            // Preview
            Group {
                if let cg = image {
                    Image(decorative: cg, scale: 1)
                        .resizable()
                        .scaledToFit()
                        .cornerRadius(12)
                } else {
                    RoundedRectangle(cornerRadius: 12)
                        .fill(.quaternary)
                        .overlay(Text(status).foregroundStyle(.secondary))
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            // Prompt + generate
            TextField("Prompt", text: $prompt, axis: .vertical)
                .lineLimit(3...6)
                .textFieldStyle(.roundedBorder)

            Button(action: generate) {
                if isWorking {
                    ProgressView()
                        .frame(maxWidth: .infinity)
                } else {
                    Text("Generate")
                        .bold()
                        .frame(maxWidth: .infinity)
                }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(isWorking || prompt.isEmpty)
        }
        .padding()
        .task {
            await loadEngineIfNeeded()
        }
    }

    // MARK: Engine lifecycle

    /// Load the engine ONCE per app launch — multi-GB I/O + GPU upload.
    /// Subsequent generations reuse the same `Engine` actor.
    private func loadEngineIfNeeded() async {
        guard engine == nil else { return }
        status = "Loading model… (multi-GB, ~10-30 s first time)"
        do {
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let models = ModelFiles(
                diffusionModel: docs.appendingPathComponent("z-image-turbo-Q3_K_M.gguf"),
                vae: docs.appendingPathComponent("ae.safetensors"),
                textEncoder: docs.appendingPathComponent("Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
            )
            engine = try Engine(models: models)
            status = "Engine ready. Tap Generate."
        } catch {
            status = "Failed to load engine: \(error)"
        }
    }

    // MARK: Generation

    private func generate() {
        guard let engine else {
            status = "Engine not ready."
            return
        }
        let promptCopy = prompt
        isWorking = true
        status = "Generating…"

        Task {
            defer { isWorking = false }
            do {
                let cg = try await engine.generate(.init(
                    prompt: promptCopy,
                    width: 1024,
                    height: 1024,
                    steps: 9,
                    cfgScale: 1.0
                ))
                image = cg
                status = "Done."
            } catch {
                status = "\(error)"
            }
        }
    }
}

#Preview {
    GenerateView()
}
