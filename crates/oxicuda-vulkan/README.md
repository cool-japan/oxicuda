# oxicuda-vulkan

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement for the COOLJAPAN ecosystem.

## Overview

`oxicuda-vulkan` is the Vulkan compute backend for OxiCUDA. It implements the `ComputeBackend` trait from `oxicuda-backend` using the Vulkan GPU API through the `ash` crate's runtime loader. No compile-time dependency on `libvulkan` is required — the library is loaded dynamically at runtime, keeping the default build fully Pure Rust.

## Features

- **Runtime Vulkan loading** — `libvulkan` is loaded via `ash::Entry::load`; no Vulkan SDK required at build time
- **SPIR-V pipeline** — Compute pipelines assembled from SPIR-V shader bytecode via the `spirv` and `pipeline` modules
- **Memory management** — Device buffer allocation, host/device transfers, and a unified memory interface
- **Command recording** — Vulkan command buffer recording and submission via the `command` module
- **Backend abstraction** — Implements `oxicuda-backend`'s `ComputeBackend` for interoperability with the rest of the OxiCUDA ecosystem
- **Cross-vendor** — Works on NVIDIA, AMD, and Intel GPUs with a Vulkan 1.2+ driver

## Platform Support

| Platform                      | Status                                  |
|-------------------------------|-----------------------------------------|
| Linux + NVIDIA/AMD/Intel      | Full (Vulkan driver 1.2+)               |
| Windows + discrete GPU        | Full (Vulkan driver 1.2+)               |
| macOS                         | `init()` returns `Err` (no native Vulkan) |

## Usage

Add to your `Cargo.toml`:
```toml
[dependencies]
oxicuda-vulkan = "0.1.3"
```

```rust
use oxicuda_vulkan::VulkanBackend;
use oxicuda_backend::ComputeBackend;

let mut backend = VulkanBackend::new();
match backend.init() {
    Ok(()) => println!("Vulkan backend ready"),
    Err(e) => println!("Vulkan not available: {e}"),
}
```

## Status

- **Version**: 0.1.3 (2026-04-17)
- **Tests**: 86 passing

## License

Apache-2.0 — © 2026 COOLJAPAN OU (Team KitaSan)
