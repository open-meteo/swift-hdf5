// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SwiftHDF5",
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "SwiftHDF5",
            targets: ["SwiftHDF5"]
        )
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .systemLibrary(
            name: "CHDF5",
            pkgConfig: "hdf5",
            providers: [.apt(["libhdf5-dev"]), .yum(["hdf5"]), .brew(["hdf5"])]
        ),
        .target(
            name: "SwiftHDF5", dependencies: ["CHDF5"]
        ),
        .testTarget(
            name: "SwiftHDF5Tests",
            dependencies: ["SwiftHDF5"]
        ),
    ]
)
