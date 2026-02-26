import CHDF5
import Foundation
import Testing

@testable import SwiftHDF5

@Suite("SwiftHDF5 Tests")
struct SwiftHDF5Tests {

    // MARK: - File Operations Tests

    @Test("Create and open HDF5 file")
    func testFileCreationAndOpening() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let testFile = tempDir.appendingPathComponent("test_create.h5").path

        // Clean up if exists
        try? FileManager.default.removeItem(atPath: testFile)

        // Create a new file
        let file = try await SwiftHDF.create(file: testFile, mode: .truncate)
        #expect(file.path == testFile)

        // Open the file
        let openedFile = try await SwiftHDF.open(file: testFile, mode: .readOnly)
        #expect(openedFile.path == testFile)

        // Clean up
        try? FileManager.default.removeItem(atPath: testFile)
    }

    // MARK: - Group Operations Tests

    @Test("Create and open groups")
    func testGroupOperations() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let testFile = tempDir.appendingPathComponent("test_groups.h5").path

        // Clean up if exists
        try? FileManager.default.removeItem(atPath: testFile)

        let file = try await SwiftHDF.create(file: testFile)

        // Create a group
        let group = try await file.createGroup("data")
        #expect(group.name == "data")

        // Create a subgroup
        let subgroup = try await group.createGroup("measurements")
        #expect(subgroup.name == "data/measurements")

        // Reopen and verify
        let reopenedFile = try await SwiftHDF.open(file: testFile, mode: .readOnly)
        let reopenedGroup = try await reopenedFile.openGroup("data")
        #expect(reopenedGroup.name == "data")

        // Clean up
        try? FileManager.default.removeItem(atPath: testFile)
    }

    // MARK: - Dataset Operations Tests

    @Test("Create and write dataset with integers")
    func testDatasetIntegerOperations() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let testFile = tempDir.appendingPathComponent("test_dataset_int.h5").path

        // Clean up if exists
        try? FileManager.default.removeItem(atPath: testFile)

        let file = try await SwiftHDF.create(file: testFile)

        // Create a 1D dataset
        let dims: [hsize_t] = [10]
        let dataspace = try await HDF5Dataspace(dimensions: dims)
        let dataset = try await file.createDataset(
            "integers",
            datatype: HDF5Datatype.int32,
            dataspace: dataspace
        )

        // Write data
        let data: [Int32] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        try await dataset.write(data)

        // Reopen and read
        let reopenedFile = try await SwiftHDF.open(file: testFile, mode: .readOnly)
        let reopenedDataset = try await reopenedFile.openDataset("integers")

        let readData: [Int32] = try await reopenedDataset.read()
        #expect(readData == data)

        // Clean up
        try? FileManager.default.removeItem(atPath: testFile)
    }

    @Test("Create and write dataset with doubles")
    func testDatasetDoubleOperations() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let testFile = tempDir.appendingPathComponent("test_dataset_double.h5").path

        // Clean up if exists
        try? FileManager.default.removeItem(atPath: testFile)

        let file = try await SwiftHDF.create(file: testFile)

        // Create a 1D dataset
        let dims: [hsize_t] = [5]
        let dataspace = try await HDF5Dataspace(dimensions: dims)
        let dataset = try await file.createDataset(
            "temperatures",
            datatype: HDF5Datatype.double,
            dataspace: dataspace
        )

        // Write data
        let data: [Double] = [20.5, 21.3, 19.8, 22.1, 20.9]
        try await dataset.write(data)

        // Reopen and read
        let reopenedFile = try await SwiftHDF.open(file: testFile, mode: .readOnly)
        let reopenedDataset = try await reopenedFile.openDataset("temperatures")

        let readData: [Double] = try await reopenedDataset.read()
        #expect(readData.count == 5)
        #expect(abs(readData[0] - 20.5) < 0.001)

        // Clean up
        try? FileManager.default.removeItem(atPath: testFile)
    }

    @Test("Create 2D dataset")
    func testTwoDimensionalDataset() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let testFile = tempDir.appendingPathComponent("test_2d_dataset.h5").path

        // Clean up if exists
        try? FileManager.default.removeItem(atPath: testFile)

        let file = try await SwiftHDF.create(file: testFile)

        // Create a 2D dataset (3x4 matrix)
        let dims: [hsize_t] = [3, 4]
        let dataspace = try await HDF5Dataspace(dimensions: dims)
        let dataset = try await file.createDataset(
            "matrix",
            datatype: HDF5Datatype.float,
            dataspace: dataspace
        )

        // Write data (row-major order)
        let data: [Float] = [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ]
        try await dataset.write(data)

        // Reopen and read
        let reopenedFile = try await SwiftHDF.open(file: testFile, mode: .readOnly)
        let reopenedDataset = try await reopenedFile.openDataset("matrix")

        let space = try await reopenedDataset.getSpace()
        let readDims = try await space.getDimensions()
        #expect(readDims.count == 2)
        #expect(readDims[0] == 3)
        #expect(readDims[1] == 4)

        let readData: [Float] = try await reopenedDataset.read()
        #expect(readData.count == 12)
        #expect(abs(readData[5] - 6.0) < 0.001)

        // Clean up
        try? FileManager.default.removeItem(atPath: testFile)
    }

    // MARK: - Attribute Tests

    @Test("Write and read file attributes")
    func testFileAttributes() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let testFile = tempDir.appendingPathComponent("test_file_attrs.h5").path

        // Clean up if exists
        try? FileManager.default.removeItem(atPath: testFile)

        let file = try await SwiftHDF.create(file: testFile)

        // Write attributes
        try await file.writeAttribute("version", value: Int32(1), datatype: HDF5Datatype.int32)
        try await file.writeAttribute("temperature", value: Double(25.5), datatype: HDF5Datatype.double)

        // Reopen and read attributes
        let reopenedFile = try await SwiftHDF.open(file: testFile, mode: .readOnly)

        let version: Int32 = try await reopenedFile.readAttribute("version")
        #expect(version == 1)

        let temperature: Double = try await reopenedFile.readAttribute("temperature")
        #expect(abs(temperature - 25.5) < 0.001)

        // Clean up
        try? FileManager.default.removeItem(atPath: testFile)
    }

    @Test("Write and read dataset attributes")
    func testDatasetAttributes() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let testFile = tempDir.appendingPathComponent("test_dataset_attrs.h5").path

        // Clean up if exists
        try? FileManager.default.removeItem(atPath: testFile)

        let file = try await SwiftHDF.create(file: testFile)

        // Create a dataset
        let dims: [hsize_t] = [5]
        let dataspace = try await HDF5Dataspace(dimensions: dims)
        let dataset = try await file.createDataset(
            "data",
            datatype: HDF5Datatype.double,
            dataspace: dataspace
        )

        // Write attributes to the dataset
        try await dataset.writeAttribute("units", value: Int32(42), datatype: HDF5Datatype.int32)
        try await dataset.writeAttribute("scale", value: Double(1.5), datatype: HDF5Datatype.double)

        // Reopen and read attributes
        let reopenedFile = try await SwiftHDF.open(file: testFile, mode: .readOnly)
        let reopenedDataset = try await reopenedFile.openDataset("data")

        let units: Int32 = try await reopenedDataset.readAttribute("units")
        #expect(units == 42)

        let scale: Double = try await reopenedDataset.readAttribute("scale")
        #expect(abs(scale - 1.5) < 0.001)

        // Clean up
        try? FileManager.default.removeItem(atPath: testFile)
    }

    // MARK: - Complex Workflow Test

    @Test("Complete workflow with groups, datasets, and attributes")
    func testCompleteWorkflow() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let testFile = tempDir.appendingPathComponent("test_workflow.h5").path

        // Clean up if exists
        try? FileManager.default.removeItem(atPath: testFile)

        // Create file
        let file = try await SwiftHDF.create(file: testFile)

        // Add file-level metadata
        try await file.writeAttribute("experiment", value: Int32(123), datatype: HDF5Datatype.int32)

        // Create groups
        let resultsGroup = try await file.createGroup("results")
        let dataGroup = try await resultsGroup.createGroup("data")

        // Create dataset in nested group
        let dims: [hsize_t] = [100]
        let dataspace = try await HDF5Dataspace(dimensions: dims)
        let dataset = try await dataGroup.createDataset(
            "measurements",
            datatype: HDF5Datatype.double,
            dataspace: dataspace
        )

        // Generate and write data
        var measurements = [Double](repeating: 0, count: 100)
        for i in 0..<100 {
            measurements[i] = Double(i) * 0.5
        }
        try await dataset.write(measurements)

        // Add metadata to dataset
        try await dataset.writeAttribute("sensor_id", value: Int32(7), datatype: HDF5Datatype.int32)
        try await dataset.writeAttribute("calibration", value: Double(1.0), datatype: HDF5Datatype.double)

        // Verify the structure
        let readFile = try await SwiftHDF.open(file: testFile, mode: .readOnly)

        let expId: Int32 = try await readFile.readAttribute("experiment")
        #expect(expId == 123)

        let readResultsGroup = try await readFile.openGroup("results")
        let readDataGroup = try await readResultsGroup.openGroup("data")
        let readDataset = try await readDataGroup.openDataset("measurements")

        let sensorId: Int32 = try await readDataset.readAttribute("sensor_id")
        #expect(sensorId == 7)

        let readMeasurements: [Double] = try await readDataset.read()
        #expect(readMeasurements.count == 100)
        #expect(abs(readMeasurements[50] - 25.0) < 0.001)

        // Clean up
        try? FileManager.default.removeItem(atPath: testFile)
    }

    // MARK: - Datatype Tests

    @Test("Test various native datatypes")
    func testNativeDatatypes() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let testFile = tempDir.appendingPathComponent("test_datatypes.h5").path

        // Clean up if exists
        try? FileManager.default.removeItem(atPath: testFile)

        let file = try await SwiftHDF.create(file: testFile)

        // Test Int8
        let space8 = try await HDF5Dataspace(dimensions: [3])
        let ds8 = try await file.createDataset(
            "int8_data",
            datatype: HDF5Datatype.int8,
            dataspace: space8
        )
        let data8: [Int8] = [1, 2, 3]
        try await ds8.write(data8)

        // Test UInt16
        let space16 = try await HDF5Dataspace(dimensions: [2])
        let ds16 = try await file.createDataset(
            "uint16_data",
            datatype: HDF5Datatype.uint16,
            dataspace: space16
        )
        let data16: [UInt16] = [100, 200]
        try await ds16.write(data16)

        // Test Int64
        let space64 = try await HDF5Dataspace(dimensions: [4])
        let ds64 = try await file.createDataset(
            "int64_data",
            datatype: HDF5Datatype.int64,
            dataspace: space64
        )
        let data64: [Int64] = [1000, 2000, 3000, 4000]
        try await ds64.write(data64)

        // Verify
        let readFile = try await SwiftHDF.open(file: testFile, mode: .readOnly)

        let readDs8 = try await readFile.openDataset("int8_data")
        let read8: [Int8] = try await readDs8.read()
        #expect(read8 == [1, 2, 3])

        // Clean up
        try? FileManager.default.removeItem(atPath: testFile)
    }
}
