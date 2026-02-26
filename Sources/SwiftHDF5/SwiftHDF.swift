import CHDF5
import Foundation

// MARK: - Error Handling

public enum HDF5Error: Error {
    case fileOpenFailed(String)
    case fileCreateFailed(String)
    case groupOpenFailed(String)
    case groupCreateFailed(String)
    case datasetOpenFailed(String)
    case datasetCreateFailed(String)
    case datasetReadFailed(String)
    case datasetWriteFailed(String)
    case dataspaceCreateFailed
    case attributeOpenFailed(String)
    case attributeCreateFailed(String)
    case attributeReadFailed(String)
    case attributeWriteFailed(String)
    case invalidDataType
    case operationFailed(String)
}

// MARK: - File Access Modes

public enum FileAccessMode: Sendable {
    case readOnly
    case readWrite
    case truncate
    case exclusive

    var cMode: UInt32 {
        switch self {
        case .readOnly: return hdf5_get_f_acc_rdonly()
        case .readWrite: return hdf5_get_f_acc_rdwr()
        case .truncate: return hdf5_get_f_acc_trunc()
        case .exclusive: return hdf5_get_f_acc_excl()
        }
    }
}

// MARK: - HDF5 Handle

final class HDF5Handle: Sendable {
    let id: hid_t
    // private var isValid: Bool = true
    private let closer: @Sendable (hid_t) -> herr_t

    init(id: hid_t, closer: @Sendable @escaping (hid_t) -> herr_t) {
        self.id = id
        self.closer = closer
    }

    deinit {
        // Important! Must capture copy of closer and id, not self
        Task { [closer, id] in
            await MainActor.shared.synchronized {
                closer(id)
            }
        }
    }

    // @discardableResult
    // private func closeInternal() async -> herr_t {
    //     await MainActor.shared.synchronized {
    //         guard isValid, id >= 0 else { return 0 }
    //         let result = closer(id)
    //         isValid = false
    //         id = -1
    //         return result
    //     }
    // }

    @discardableResult
    func close() async throws -> Bool {
        // Important! Must capture copy of closer and id, not self
        Task { [closer, id] in
            await MainActor.shared.synchronized {
                closer(id)
            }
        }
        // FIXME: This is not necessarily correct....
        return true
    }
}

// MARK: - Protocols

public protocol HDF5Object: Sendable {
    var id: hid_t { get }
    var name: String { get }
}

protocol HDF5Closable: HDF5Object {
    var handle: HDF5Handle { get }
}

extension HDF5Closable {
    @discardableResult
    public func close() async throws -> Bool {
        try await handle.close()
    }
}

public protocol HDF5Container: HDF5Object {
    var fileReference: HDF5File? { get }
}

extension HDF5Container {
    public func openGroup(_ groupName: String) async throws -> HDF5Group {
        try await MainActor.shared.synchronized {
            guard let file = fileReference else {
                throw HDF5Error.groupOpenFailed(groupName)
            }
            let groupId = groupName.withCString {
                H5Gopen2(id, $0, hdf5_get_p_default())
            }
            guard groupId >= 0 else { throw HDF5Error.groupOpenFailed(groupName) }
            let fullName = name.isEmpty ? groupName : "\(name)/\(groupName)"
            return HDF5Group(name: fullName, id: groupId, file: file)
        }
    }

    public func createGroup(_ groupName: String) async throws -> HDF5Group {
        try await MainActor.shared.synchronized {
            guard let file = fileReference else {
                throw HDF5Error.groupCreateFailed(groupName)
            }
            let groupId = groupName.withCString {
                H5Gcreate2(
                    id,
                    $0,
                    hdf5_get_p_default(),
                    hdf5_get_p_default(),
                    hdf5_get_p_default()
                )
            }
            guard groupId >= 0 else { throw HDF5Error.groupCreateFailed(groupName) }
            let fullName = name.isEmpty ? groupName : "\(name)/\(groupName)"
            return HDF5Group(name: fullName, id: groupId, file: file)
        }
    }

    public func openDataset(_ datasetName: String) async throws -> HDF5Dataset {
        try await MainActor.shared.synchronized {
            let datasetId = datasetName.withCString {
                H5Dopen2(id, $0, hdf5_get_p_default())
            }
            guard datasetId >= 0 else { throw HDF5Error.datasetOpenFailed(datasetName) }
            let fullName = name.isEmpty ? datasetName : "\(name)/\(datasetName)"
            return HDF5Dataset(name: fullName, id: datasetId)
        }
    }

    public func createDataset(
        _ datasetName: String,
        datatype: hid_t,
        dataspace: HDF5Dataspace
    ) async throws -> HDF5Dataset {
        try await MainActor.shared.synchronized {
            let datasetId = datasetName.withCString {
                H5Dcreate2(
                    id,
                    $0,
                    datatype,
                    dataspace.id,
                    hdf5_get_p_default(),
                    hdf5_get_p_default(),
                    hdf5_get_p_default()
                )
            }
            guard datasetId >= 0 else { throw HDF5Error.datasetCreateFailed(datasetName) }
            let fullName = name.isEmpty ? datasetName : "\(name)/\(datasetName)"
            return HDF5Dataset(name: fullName, id: datasetId)
        }
    }
}

public protocol HDF5AttributeContainer: HDF5Object {
    func writeAttribute<T: Sendable>(_ name: String, value: T, datatype: hid_t) async throws
    func readAttribute<T: Sendable>(_ name: String) async throws -> T
}

extension HDF5AttributeContainer {
    public func writeAttribute<T: Sendable>(_ name: String, value: T, datatype: hid_t) async throws {
        try await MainActor.shared.synchronized {
            let dataspaceId = H5Screate(hdf5_get_s_scalar())
            guard dataspaceId >= 0 else { throw HDF5Error.dataspaceCreateFailed }
            defer { H5Sclose(dataspaceId) }

            let attrId = name.withCString {
                H5Acreate2(
                    id,
                    $0,
                    datatype,
                    dataspaceId,
                    hdf5_get_p_default(),
                    hdf5_get_p_default()
                )
            }
            guard attrId >= 0 else { throw HDF5Error.attributeCreateFailed(name) }
            defer { H5Aclose(attrId) }

            var mutableValue = value
            let res = withUnsafePointer(to: &mutableValue) { ptr in
                H5Awrite(attrId, datatype, ptr)
            }
            guard res >= 0 else { throw HDF5Error.attributeWriteFailed(name) }
        }
    }

    public func readAttribute<T: Sendable>(_ name: String) async throws -> T {
        try await MainActor.shared.synchronized {
            let attrId = name.withCString {
                H5Aopen(id, $0, hdf5_get_p_default())
            }
            guard attrId >= 0 else { throw HDF5Error.attributeOpenFailed(name) }
            defer { H5Aclose(attrId) }

            let typeId = H5Aget_type(attrId)
            guard typeId >= 0 else { throw HDF5Error.invalidDataType }
            defer { H5Tclose(typeId) }

            // Special handling for String types
            if T.self == String.self {
                let size = H5Tget_size(typeId)
                guard size > 0 else { return "" as! T }

                var buffer = [UInt8](repeating: 0, count: size)
                let res = buffer.withUnsafeMutableBufferPointer { ptr in
                    H5Aread(attrId, typeId, ptr.baseAddress)
                }
                guard res >= 0 else { throw HDF5Error.attributeReadFailed(name) }

                let len = buffer.firstIndex(of: 0) ?? buffer.count
                let str =
                    String(bytes: buffer[..<len], encoding: .utf8)?
                    .trimmingCharacters(in: .whitespaces) ?? ""
                return str as! T
            }

            // Generic handling for fixed-size Numeric/POD types
            return try withUnsafeTemporaryAllocation(of: T.self, capacity: 1) { buffer -> T in
                guard H5Aread(attrId, typeId, buffer.baseAddress) >= 0 else {
                    throw HDF5Error.attributeReadFailed(name)
                }
                return buffer[0]
            }
        }
    }
}

// MARK: - Main SwiftHDF Interface

public struct SwiftHDF {
    public static func open(file: String, mode: FileAccessMode = .readOnly) async throws -> HDF5File {
        try await MainActor.shared.synchronized {
            _ = H5open()
            let fileId = file.withCString { cPath in
                H5Fopen(cPath, mode.cMode, hdf5_get_p_default())
            }
            guard fileId >= 0 else { throw HDF5Error.fileOpenFailed(file) }
            return HDF5File(path: file, id: fileId)
        }
    }

    public static func create(file: String, mode: FileAccessMode = .truncate) async throws -> HDF5File {
        try await MainActor.shared.synchronized {
            _ = H5open()
            let fileId = file.withCString { cPath in
                H5Fcreate(
                    cPath,
                    mode.cMode,
                    hdf5_get_p_default(),
                    hdf5_get_p_default()
                )
            }
            guard fileId >= 0 else { throw HDF5Error.fileCreateFailed(file) }
            return HDF5File(path: file, id: fileId)
        }
    }
}

// MARK: - HDF5 File

public struct HDF5File: HDF5Container, HDF5AttributeContainer {
    public let path: String
    private let handle: HDF5Handle

    public var id: hid_t { handle.id }
    public var name: String { "" }
    public var fileReference: HDF5File? { self }

    init(path: String, id: hid_t) {
        self.path = path
        self.handle = HDF5Handle(id: id, closer: H5Fclose)
    }

    @discardableResult
    public func close() async throws -> Bool {
        try await handle.close()
    }
}

// MARK: - HDF5 Group

public struct HDF5Group: HDF5Container, HDF5Closable, HDF5AttributeContainer {
    public let name: String
    internal let handle: HDF5Handle
    public let fileReference: HDF5File?

    public var id: hid_t { handle.id }

    init(name: String, id: hid_t, file: HDF5File) {
        self.name = name
        self.handle = HDF5Handle(id: id, closer: H5Gclose)
        self.fileReference = file
    }
}

// MARK: - HDF5 Dataspace

public struct HDF5Dataspace: HDF5Object, HDF5Closable {
    public let name: String = ""
    internal let handle: HDF5Handle

    public var id: hid_t { handle.id }

    public init(dimensions: [hsize_t]) async throws {
        let spaceId = await MainActor.shared.synchronized {
            dimensions.withUnsafeBufferPointer { ptr in
                H5Screate_simple(Int32(dimensions.count), ptr.baseAddress, nil)
            }
        }
        guard spaceId >= 0 else { throw HDF5Error.dataspaceCreateFailed }
        self.handle = HDF5Handle(id: spaceId, closer: H5Sclose)
    }

    public init() async throws {
        let spaceId = await MainActor.shared.synchronized {
            H5Screate(hdf5_get_s_scalar())
        }
        guard spaceId >= 0 else { throw HDF5Error.dataspaceCreateFailed }
        self.handle = HDF5Handle(id: spaceId, closer: H5Sclose)
    }

    init(id: hid_t) {
        self.handle = HDF5Handle(id: id, closer: H5Sclose)
    }

    public func getDimensions() async throws -> [hsize_t] {
        try await MainActor.shared.synchronized {
            let ndims = H5Sget_simple_extent_ndims(id)
            guard ndims >= 0 else { throw HDF5Error.operationFailed("Failed to get dimensions") }

            var dims = [hsize_t](repeating: 0, count: Int(ndims))
            let res = dims.withUnsafeMutableBufferPointer { ptr in
                H5Sget_simple_extent_dims(id, ptr.baseAddress, nil)
            }
            guard res >= 0 else { throw HDF5Error.operationFailed("Failed to get dimensions") }
            return dims
        }
    }
}

// MARK: - HDF5 Dataset

public struct HDF5Dataset: HDF5Object, HDF5Closable, HDF5AttributeContainer {
    public let name: String
    internal let handle: HDF5Handle

    public var id: hid_t { handle.id }

    init(name: String, id: hid_t) {
        self.name = name
        self.handle = HDF5Handle(id: id, closer: H5Dclose)
    }

    public func getSpace() async throws -> HDF5Dataspace {
        try await MainActor.shared.synchronized {
            let spaceId = H5Dget_space(id)
            guard spaceId >= 0 else { throw HDF5Error.operationFailed("Failed to get dataspace") }
            return HDF5Dataspace(id: spaceId)
        }
    }

    public func read<T: Sendable>() async throws -> [T] where T: Numeric {
        try await MainActor.shared.synchronized {
            let spaceId = H5Dget_space(id)
            guard spaceId >= 0 else { throw HDF5Error.operationFailed("Failed to get dataspace") }
            defer { H5Sclose(spaceId) }

            let ndims = H5Sget_simple_extent_ndims(spaceId)
            guard ndims >= 0 else { throw HDF5Error.operationFailed("Failed to get dimensions") }

            var dims = [hsize_t](repeating: 0, count: Int(ndims))
            let dimRes = dims.withUnsafeMutableBufferPointer { ptr in
                H5Sget_simple_extent_dims(spaceId, ptr.baseAddress, nil)
            }
            guard dimRes >= 0 else { throw HDF5Error.operationFailed("Failed to get dimensions") }

            let count = dims.reduce(1, *)
            var buffer = [T](repeating: 0, count: Int(count))

            let typeId = H5Dget_type(id)
            guard typeId >= 0 else { throw HDF5Error.invalidDataType }
            defer { H5Tclose(typeId) }

            let res = buffer.withUnsafeMutableBufferPointer { ptr in
                H5Dread(
                    id,
                    typeId,
                    hdf5_get_s_all(),
                    hdf5_get_s_all(),
                    hdf5_get_p_default(),
                    ptr.baseAddress
                )
            }
            guard res >= 0 else { throw HDF5Error.datasetReadFailed(name) }
            return buffer
        }
    }

    public func write<T: Sendable>(_ data: [T]) async throws {
        try await MainActor.shared.synchronized {
            let typeId = H5Dget_type(id)
            guard typeId >= 0 else { throw HDF5Error.invalidDataType }
            defer { H5Tclose(typeId) }

            let res = data.withUnsafeBufferPointer { ptr in
                H5Dwrite(
                    id,
                    typeId,
                    hdf5_get_s_all(),
                    hdf5_get_s_all(),
                    hdf5_get_p_default(),
                    ptr.baseAddress
                )
            }
            guard res >= 0 else { throw HDF5Error.datasetWriteFailed(name) }
        }
    }
}

// MARK: - HDF5 Datatype

public struct HDF5Datatype {
    public static var int8: hid_t { hdf5_get_native_int8() }
    public static var int16: hid_t { hdf5_get_native_int16() }
    public static var int32: hid_t { hdf5_get_native_int32() }
    public static var int64: hid_t { hdf5_get_native_int64() }
    public static var uint8: hid_t { hdf5_get_native_uint8() }
    public static var uint16: hid_t { hdf5_get_native_uint16() }
    public static var uint32: hid_t { hdf5_get_native_uint32() }
    public static var uint64: hid_t { hdf5_get_native_uint64() }
    public static var float: hid_t { hdf5_get_native_float() }
    public static var double: hid_t { hdf5_get_native_double() }
    public static var char: hid_t { hdf5_get_native_char() }
}
