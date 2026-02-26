# SwiftHDF5
![Swift 6](https://img.shields.io/badge/Swift-6-orange.svg) ![SPM](https://img.shields.io/badge/SPM-compatible-green.svg) ![Platforms](https://img.shields.io/badge/Platforms-macOS%20Linux-green.svg) [![Test](https://github.com/open-meteo/swift-hdf5/actions/workflows/lint_test.yml/badge.svg)](https://github.com/open-meteo/swift-hdf5/actions/workflows/lint_test.yml)

A Swift wrapper for the HDF5 C library, providing a modern, type-safe interface for reading and writing HDF5 files.

## Overview

SwiftHDF5 provides a Swift-friendly API for working with HDF5 files, similar to what SwiftNetCDF does for NetCDF. It wraps the HDF5 C library with Swift classes and protocols, making it easier to work with hierarchical data formats in Swift applications.

## Requirements

- Swift 6.0 or later
- HDF5 library installed on your system

### Swift Package Manager

Add SwiftHDF5 to your `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/open-meteo/swift-hdf5.git", branch: "main")
],
targets: [
    .target(
        name: "YourTarget",
        dependencies: [
            .product(name: "SwiftHDF5", package: "swift-hdf5"),
        ]
    )
]
```

## Resource Management

SwiftHDF5 classes implement automatic cleanup via `deinit`, so you don't need to manually close resources in most cases. However, you can explicitly close resources if needed:

```swift
let file = try SwiftHDF.create(file: "data.h5")
// Work with file...
file.close()  // Optional explicit close
// Or just let it go out of scope - deinit will handle cleanup
```

## Development

### Building

```bash
swift build
```

### Testing

```bash
swift test
```

### Running Tests with HDF5 Installed

Ensure HDF5 is installed and available on your system before running tests.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Acknowledgments

- Inspired by SwiftNetCDF
- Built on top of the HDF5 C library
- Thanks to the HDF Group for HDF5

## Related Projects

- [HDF5](https://www.hdfgroup.org/solutions/hdf5/) - Hierarchical Data Format version 5
