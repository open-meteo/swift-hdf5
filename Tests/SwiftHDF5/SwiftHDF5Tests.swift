import CHDF5
import Foundation
import Testing

@testable import SwiftHDF5

@Suite("SwiftHDF5 Tests")
struct SwiftHDF5Tests {

    let hdf5 = HDF5.shared

    // MARK: - File Operations Tests

    @Test("Create and open HDF5 file")
    func testFileCreationAndOpening() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let testFile = tempDir.appendingPathComponent("test_create.h5").path
        try? FileManager.default.removeItem(atPath: testFile)

        let file = try await hdf5.createFile(testFile, mode: .truncate)
        #expect(file.path == testFile)

        let openedFile = try await hdf5.openFile(testFile, mode: .readOnly)
        #expect(openedFile.path == testFile)

        try? FileManager.default.removeItem(atPath: testFile)
    }

    // MARK: - Group Operations Tests

    @Test("Create and open groups")
    func testGroupOperations() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let testFile = tempDir.appendingPathComponent("test_groups.h5").path
        try? FileManager.default.removeItem(atPath: testFile)

        let file = try await hdf5.createFile(testFile)

        let group = try await file.createGroup("data")
        #expect(group.name == "data")

        let subgroup = try await group.createGroup("measurements")
        #expect(subgroup.name == "data/measurements")

        let reopenedFile = try await hdf5.openFile(testFile, mode: .readOnly)
        let reopenedGroup = try await reopenedFile.openGroup("data")
        #expect(reopenedGroup.name == "data")

        try? FileManager.default.removeItem(atPath: testFile)
    }

    // MARK: - Dataset Operations Tests

    @Test("Create and write dataset with integers")
    func testDatasetIntegerOperations() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let testFile = tempDir.appendingPathComponent("test_dataset_int.h5").path
        try? FileManager.default.removeItem(atPath: testFile)

        let file = try await hdf5.createFile(testFile)
        let dataspace = try await hdf5.createDataspace(dimensions: [10])
        let dataset = try await file.createDataset(
            "integers",
            datatype: HDF5Datatype.int32,
            dataspace: dataspace
        )
        let data: [Int32] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        try await dataset.writeDataset(data: data)

        let reopenedFile = try await hdf5.openFile(testFile, mode: .readOnly)
        let reopenedDataset = try await reopenedFile.openDataset("integers")

        let readData: [Int32] = try await hdf5.readDataset(reopenedDataset)
        #expect(readData == data)

        // Also test read(into:)
        var preallocated = [Int32](repeating: 0, count: 10)
        try await hdf5.readDataset(reopenedDataset, into: &preallocated)
        #expect(preallocated == data)

        try? FileManager.default.removeItem(atPath: testFile)
    }

    @Test("Create and write dataset with doubles")
    func testDatasetDoubleOperations() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let testFile = tempDir.appendingPathComponent("test_dataset_double.h5").path
        try? FileManager.default.removeItem(atPath: testFile)

        let file = try await hdf5.createFile(testFile)
        let dataspace = try await hdf5.createDataspace(dimensions: [5])
        let dataset = try await file.createDataset(
            "temperatures",
            datatype: HDF5Datatype.double,
            dataspace: dataspace
        )

        let data: [Double] = [20.5, 21.3, 19.8, 22.1, 20.9]
        try await dataset.writeDataset(data: data)

        let reopenedFile = try await hdf5.openFile(testFile, mode: .readOnly)
        let reopenedDataset = try await reopenedFile.openDataset("temperatures")

        let readData: [Double] = try await hdf5.readDataset(reopenedDataset)
        #expect(readData.count == 5)
        #expect(abs(readData[0] - 20.5) < 0.001)

        try? FileManager.default.removeItem(atPath: testFile)
    }

    @Test("Create 2D dataset")
    func testTwoDimensionalDataset() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let testFile = tempDir.appendingPathComponent("test_2d_dataset.h5").path
        try? FileManager.default.removeItem(atPath: testFile)

        let file = try await hdf5.createFile(testFile)
        let dataspace = try await hdf5.createDataspace(dimensions: [3, 4])
        let dataset = try await file.createDataset(
            "matrix",
            datatype: HDF5Datatype.float,
            dataspace: dataspace
        )
        let data: [Float] = [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ]
        try await dataset.writeDataset(data: data)

        let reopenedFile = try await hdf5.openFile(testFile, mode: .readOnly)
        let reopenedDataset = try await reopenedFile.openDataset("matrix")

        let space = try await reopenedDataset.getDatasetSpace()
        let readDims = try await space.getDimensions()
        #expect(readDims.count == 2)
        #expect(readDims[0] == 3)
        #expect(readDims[1] == 4)
        
        var readData = [Float](repeating: 0, count: 12)
        try await hdf5.readDataset(reopenedDataset, into: &readData)
        #expect(readData.count == 12)
        #expect(abs(readData[5] - 6.0) < 0.001)

        try? FileManager.default.removeItem(atPath: testFile)
    }

    // MARK: - Attribute Tests

    @Test("Write and read file attributes")
    func testFileAttributes() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let testFile = tempDir.appendingPathComponent("test_file_attrs.h5").path
        try? FileManager.default.removeItem(atPath: testFile)

        let file = try await hdf5.createFile(testFile)
        try await hdf5.writeAttribute("version", on: file, value: Int32(1), datatype: HDF5Datatype.int32)
        try await hdf5.writeAttribute("temperature", on: file, value: Double(25.5), datatype: HDF5Datatype.double)

        let reopenedFile = try await hdf5.openFile(testFile, mode: .readOnly)
        let version: Int32 = try await hdf5.readAttribute("version", from: reopenedFile)
        #expect(version == 1)
        let temperature: Double = try await hdf5.readAttribute("temperature", from: reopenedFile)
        #expect(abs(temperature - 25.5) < 0.001)

        try? FileManager.default.removeItem(atPath: testFile)
    }

    @Test("Write and read dataset attributes")
    func testDatasetAttributes() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let testFile = tempDir.appendingPathComponent("test_dataset_attrs.h5").path
        try? FileManager.default.removeItem(atPath: testFile)

        let file = try await hdf5.createFile(testFile)
        let dataspace = try await hdf5.createDataspace(dimensions: [5])
        let dataset = try await file.createDataset(
            "data",
            datatype: HDF5Datatype.double,
            dataspace: dataspace
        )

        try await hdf5.writeAttribute("units", on: dataset, value: Int32(42), datatype: HDF5Datatype.int32)
        try await hdf5.writeAttribute("scale", on: dataset, value: Double(1.5), datatype: HDF5Datatype.double)

        let reopenedFile = try await hdf5.openFile(testFile, mode: .readOnly)
        let reopenedDataset = try await reopenedFile.openDataset("data")

        let units: Int32 = try await hdf5.readAttribute("units", from: reopenedDataset)
        #expect(units == 42)
        let scale: Double = try await hdf5.readAttribute("scale", from: reopenedDataset)
        #expect(abs(scale - 1.5) < 0.001)

        try? FileManager.default.removeItem(atPath: testFile)
    }

    // MARK: - Complex Workflow Test

    @Test("Complete workflow with groups, datasets, and attributes")
    func testCompleteWorkflow() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let testFile = tempDir.appendingPathComponent("test_workflow.h5").path
        try? FileManager.default.removeItem(atPath: testFile)

        let file = try await hdf5.createFile(testFile)
        try await hdf5.writeAttribute("experiment", on: file, value: Int32(123), datatype: HDF5Datatype.int32)

        let resultsGroup = try await file.createGroup("results")
        let dataGroup = try await resultsGroup.createGroup("data")

        let dataspace = try await hdf5.createDataspace(dimensions: [100])
        let dataset = try await dataGroup.createDataset(
            "measurements",
            datatype: HDF5Datatype.double,
            dataspace: dataspace
        )

        var measurements = [Double](repeating: 0, count: 100)
        for i in 0..<100 { measurements[i] = Double(i) * 0.5 }
        try await dataset.writeDataset(data: measurements)

        try await hdf5.writeAttribute("sensor_id", on: dataset, value: Int32(7), datatype: HDF5Datatype.int32)
        try await hdf5.writeAttribute("calibration", on: dataset, value: Double(1.0), datatype: HDF5Datatype.double)

        let readFile = try await hdf5.openFile(testFile, mode: .readOnly)
        let expId: Int32 = try await hdf5.readAttribute("experiment", from: readFile)
        #expect(expId == 123)

        let readResultsGroup = try await readFile.openGroup("results")
        let readDataGroup = try await readResultsGroup.openGroup("data")
        let readDataset = try await readDataGroup.openDataset("measurements")

        let sensorId: Int32 = try await hdf5.readAttribute("sensor_id", from: readDataset)
        #expect(sensorId == 7)

        var readMeasurements = [Double](repeating: 0, count: 100)
        try await hdf5.readDataset(readDataset, into: &readMeasurements)
        #expect(readMeasurements.count == 100)
        #expect(abs(readMeasurements[50] - 25.0) < 0.001)

        try? FileManager.default.removeItem(atPath: testFile)
    }

    // MARK: - Datatype Tests

    @Test("Test various native datatypes")
    func testNativeDatatypes() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let testFile = tempDir.appendingPathComponent("test_datatypes.h5").path
        try? FileManager.default.removeItem(atPath: testFile)

        let file = try await hdf5.createFile(testFile)

        let space8 = try await hdf5.createDataspace(dimensions: [3])
        let ds8 = try await file.createDataset("int8_data", datatype: HDF5Datatype.int8, dataspace: space8)
        try await ds8.writeDataset(data: [Int8(1), Int8(2), Int8(3)])

        let space16 = try await hdf5.createDataspace(dimensions: [2])
        let ds16 = try await file.createDataset(
            "uint16_data",
            datatype: HDF5Datatype.uint16,
            dataspace: space16
        )
        try await ds16.writeDataset(data: [UInt16(100), UInt16(200)])

        let space64 = try await hdf5.createDataspace(dimensions: [4])
        let ds64 = try await file.createDataset(
            "int64_data",
            datatype: HDF5Datatype.int64,
            dataspace: space64
        )
        try await ds64.writeDataset(data: [Int64(1000), Int64(2000), Int64(3000), Int64(4000)])

        let readFile = try await hdf5.openFile(testFile, mode: .readOnly)
        let readDs8 = try await readFile.openDataset("int8_data")
        var read8 = [Int8](repeating: 0, count: 3)
        try await hdf5.readDataset(readDs8, into: &read8)
        #expect(read8 == [1, 2, 3])

        try? FileManager.default.removeItem(atPath: testFile)
    }
}
