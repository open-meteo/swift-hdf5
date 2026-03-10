# SwiftHDF5
![Swift 6](https://img.shields.io/badge/Swift-6-orange.svg) ![SPM](https://img.shields.io/badge/SPM-compatible-green.svg) ![Platforms](https://img.shields.io/badge/Platforms-macOS%20Linux-green.svg) [![Test](https://github.com/open-meteo/swift-hdf5/actions/workflows/lint_test.yml/badge.svg)](https://github.com/open-meteo/swift-hdf5/actions/workflows/lint_test.yml)

A Swift wrapper for the HDF5 C library, providing a modern, type-safe interface for reading and writing HDF5 files.

## Overview

SwiftHDF5 provides a Swift-friendly API for working with HDF5 files, similar to what SwiftNetCDF does for NetCDF. It wraps the HDF5 C library with Swift classes and protocols, making it easier to work with hierarchical data formats in Swift applications.

## Requirements

- Swift 6.0 or later
- HDF5 library installed on your system

## Installation

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

## Usage

### Creating and writing a file

```swift
// Create a new file (truncates any existing file at that path)
let file = try await HDF5.createFile("/tmp/data.h5")

// Create a flat 1-D dataset of 10 Int32 values
let space   = try await HDF5.createDataspace(dimensions: [10])
let dataset = try await file.createDataset(
    "integers",
    datatype: HDF5Datatype.int32,
    dataspace: space
)
try await dataset.writeDataset(data: [Int32](0..<10))

// Attach metadata attributes to the dataset
try await dataset.writeAttribute("unit", value: "counts")
try await dataset.writeAttribute("scale_factor", value: Double(0.01))

// Organise data into groups
let sensors  = try await file.createGroup("sensors")
let tempData = try await sensors.createGroup("temperature")

let tempSpace   = try await HDF5.createDataspace(dimensions: [5])
let tempDataset = try await tempData.createDataset(
    "readings",
    datatype: HDF5Datatype.double,
    dataspace: tempSpace
)
try await tempDataset.writeDataset(data: [20.5, 21.3, 19.8, 22.1, 20.9])
```

### Reading an existing file

```swift
// Open an existing file read-only (the default)
let file = try await HDF5.openFile("/tmp/data.h5")

// Read back a top-level dataset
let dataset: HDF5Dataset = try await file.openDataset("integers")
let values: [Int32]      = try await dataset.readDataset()

// Read attributes
let unit: String        = try await dataset.readAttribute("unit")
let scale: Double       = try await dataset.readAttribute("scale_factor")

// Navigate the group hierarchy
let sensors     = try await file.openGroup("sensors")
let tempData    = try await sensors.openGroup("temperature")
let tempDataset = try await tempData.openDataset("readings")

// Inspect the shape before reading
let dims = try await tempDataset.space.dimensions  // → [5]
let readings: [Double] = try await tempDataset.readDataset()
```

### Multi-dimensional datasets

```swift
// Create a 3×4 matrix dataset
let space   = try await HDF5.createDataspace(dimensions: [3, 4])
let dataset = try await file.createDataset(
    "matrix",
    datatype: HDF5Datatype.float,
    dataspace: space
)

// Data is stored and returned in row-major (C) order
let data: [Float] = [
    1,  2,  3,  4,
    5,  6,  7,  8,
    9, 10, 11, 12,
]
try await dataset.writeDataset(data: data)

// Read back and reshape using the stored dimensions
let readData: [Float]    = try await dataset.readDataset()
let dims: [UInt64]       = try await dataset.space.dimensions  // → [3, 4]
```

### Re-using a pre-allocated read buffer

When reading large datasets in a tight loop you can avoid repeated allocations
by passing a buffer for SwiftHDF5 to write into:

```swift
var buffer = [Float](repeating: 0, count: 1024)
buffer = try await dataset.readDataset(reusing: consume buffer)
```

## Thread Safety

The HDF5 C library is **not thread-safe by default**. SwiftHDF5 addresses this
by serialising every call to the C library through a single internal
`DispatchQueue`. Every `async` method suspends the caller and resumes on that
queue, so concurrent calls from multiple Swift `Task`s are safe — they are
queued and executed one at a time.

You do not need to add any additional synchronisation when using this library
from multiple tasks or actors.

## Resource Management

SwiftHDF5 classes implement automatic cleanup via `deinit`, so you do not need
to manually close resources. Files, groups, datasets, and dataspaces are all
closed as soon as the last reference to the Swift object goes out of scope:

```swift
do {
    let file = try await HDF5.createFile("/tmp/data.h5")
    // ... work with file ...
} // file is closed here automatically
```

## Supported Types

The following Swift types can be used as dataset element types and attribute
values out of the box:

| Swift type | HDF5 datatype constant  |
|------------|-------------------------|
| `Int8`     | `HDF5Datatype.int8`     |
| `Int16`    | `HDF5Datatype.int16`    |
| `Int32`    | `HDF5Datatype.int32`    |
| `Int64`    | `HDF5Datatype.int64`    |
| `UInt8`    | `HDF5Datatype.uint8`    |
| `UInt16`   | `HDF5Datatype.uint16`   |
| `UInt32`   | `HDF5Datatype.uint32`   |
| `UInt64`   | `HDF5Datatype.uint64`   |
| `Float`    | `HDF5Datatype.float`    |
| `Double`   | `HDF5Datatype.double`   |
| `String`   | variable-length string  |

## Development

### Building

```bash
swift build
```

### Testing

```bash
swift test
```

Ensure HDF5 is installed and available on your system before running tests.

### Installing HDF5

**macOS**
```bash
brew install hdf5
```

**Ubuntu / Debian**
```bash
sudo apt-get install libhdf5-dev
```

**Fedora / RHEL**
```bash
sudo yum install hdf5
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Related Projects

- [HDF5](https://www.hdfgroup.org/solutions/hdf5/) - Hierarchical Data Format version 5
