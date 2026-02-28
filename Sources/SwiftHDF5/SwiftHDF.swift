import CHDF5
import Foundation

// MARK: - Error Handling

public enum HDF5Error: Error {
    case fileOpenFailed(String)
    case fileCreateFailed(String)
    case fileCloseFailed
    case groupOpenFailed(String)
    case groupCreateFailed(String)
    case groupCloseFailed
    case datasetOpenFailed(String)
    case datasetCreateFailed(String)
    case datasetReadFailed(String)
    case datasetWriteFailed(String)
    case datasetCloseFailed
    case dataspaceCreateFailed
    case dataspaceCloseFailed
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

public final class HDF5FileRef: Sendable {
    public let path: String
    public let id: hid_t
    
    init(path: String, id: hid_t) {
        self.path = path
        self.id = id
    }
    
    deinit {
        Task { [id] in
            try? await HDF5.shared.h5Fclose(id)
        }
    }
}

public final class HDF5GroupRef: Sendable {
    public let name: String
    public let id: hid_t
    public let fileId: hid_t
    public let parent: HDF5Parent
    
    init(name: String, id: hid_t, fileId: hid_t, parent: some HDF5Parent) {
        self.name = name
        self.id = id
        self.fileId = fileId
        self.parent = parent
    }
    
    deinit {
        Task { [id] in
            try? await HDF5.shared.h5Gclose(id)
        }
    }
}

public final class HDF5DatasetRef: Sendable {
    public let name: String
    public let id: hid_t
    
    init(name: String, id: hid_t) {
        self.name = name
        self.id = id
    }
    
    public func writeDataset<T: Sendable>(data: [T]) async throws {
        try await HDF5.shared.h5Dwrite(dataset: self.id, data: data)
    }
    
    public func getDatasetSpace() async throws -> HDF5DataspaceRef {
        let spaceId = try await HDF5.shared.h5Dget_space(dataset: id)
        return HDF5DataspaceRef(name: "", id: spaceId)
    }
    
    deinit {
        Task { [id] in
            try? await HDF5.shared.h5Dclose(id)
        }
    }
}

public final class HDF5DataspaceRef: Sendable {
    public let name: String
    public let id: hid_t
    
    init(name: String, id: hid_t) {
        self.name = name
        self.id = id
    }
    
    public func getDimensions() async throws -> [hsize_t] {
        return try await HDF5.shared.h5Sget_simple_extent_dims(space_id: id)
    }
    
    deinit {
        Task { [id] in
            try? await HDF5.shared.h5Sclose(id)
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

// MARK: - The single actor that serializes ALL HDF5 C calls

public actor HDF5 {
    /// Single gateway to all HDF5 C API calls.
    ///
    /// There must only ever be one instance of this actor in the process.
    /// HDF5's C library maintains global internal state and is not thread-safe.
    /// All calls are serialised through this actor's executor. Creating a second
    /// instance would allow two threads to call into the C library concurrently,
    /// which causes crashes and data corruption.
    public static let shared = HDF5()

    private init() {
        H5open()
    }
    
    deinit {
        H5close()
    }

    // MARK: - File operations

    public func createFile(_ path: String, mode: FileAccessMode = .truncate) throws -> HDF5FileRef {
        let fileId = path.withCString { cPath in
            H5Fcreate(cPath, mode.cMode, hdf5_get_p_default(), hdf5_get_p_default())
        }
        guard fileId >= 0 else { throw HDF5Error.fileCreateFailed(path) }
        return HDF5FileRef(path: path, id: fileId)
    }

    public func openFile(_ path: String, mode: FileAccessMode = .readOnly) throws -> HDF5FileRef {
        let fileId = path.withCString { cPath in
            H5Fopen(cPath, mode.cMode, hdf5_get_p_default())
        }
        guard fileId >= 0 else { throw HDF5Error.fileOpenFailed(path) }
        return HDF5FileRef(path: path, id: fileId)
    }
    
    fileprivate func h5Fclose(_ id: hid_t) throws {
        guard H5Fclose(id) >= 0 else {
            throw HDF5Error.fileCloseFailed
        }
    }

    // MARK: - Group operations
    
    fileprivate func h5Gcreate2(_ name: String, _ parentId: hid_t) throws -> hid_t {
        let groupId = name.withCString {
            H5Gcreate2(
                parentId,
                $0,
                hdf5_get_p_default(),
                hdf5_get_p_default(),
                hdf5_get_p_default()
            )
        }
        guard groupId >= 0 else { throw HDF5Error.groupCreateFailed(name) }
        return groupId
    }
    
    fileprivate func h5Gopen2(_ name: String, _ parentId: hid_t) throws -> hid_t {
        let groupId = name.withCString {
            H5Gopen2(parentId, $0, hdf5_get_p_default())
        }
        guard groupId >= 0 else { throw HDF5Error.groupOpenFailed(name) }
        return groupId
    }
    
    fileprivate func h5Gclose(_ id: hid_t) throws {
        guard H5Gclose(id) >= 0 else {
            throw HDF5Error.groupCloseFailed
        }
    }

    // MARK: - Dataspace operations

    public func createDataspace(dimensions: [hsize_t]) throws -> HDF5DataspaceRef {
        let spaceId = dimensions.withUnsafeBufferPointer { ptr in
            H5Screate_simple(Int32(dimensions.count), ptr.baseAddress, nil)
        }
        guard spaceId >= 0 else { throw HDF5Error.dataspaceCreateFailed }
        return HDF5DataspaceRef(name: "", id: spaceId)
    }
    
//    fileprivate func h5Screate_simple(dimensions: [hsize_t]) -> hid_t {
//        return dimensions.withUnsafeBufferPointer { ptr in
//            H5Screate_simple(Int32(dimensions.count), ptr.baseAddress, nil)
//        }
//    }
    
    fileprivate func h5Sget_simple_extent_dims(space_id: hid_t) throws -> [hsize_t]  {
        let ndims = H5Sget_simple_extent_ndims(space_id)
        guard ndims >= 0 else { throw HDF5Error.operationFailed("Failed to get dimensions") }

        var dims = [hsize_t](repeating: 0, count: Int(ndims))
        let res = dims.withUnsafeMutableBufferPointer { ptr in
            H5Sget_simple_extent_dims(space_id, ptr.baseAddress, nil)
        }
        guard res >= 0 else { throw HDF5Error.operationFailed("Failed to get dimensions") }
        return dims
    }
    
    fileprivate func h5Sclose(_ id: hid_t) throws {
        guard H5Sclose(id) >= 0 else {
            throw HDF5Error.dataspaceCloseFailed
        }
    }

    // MARK: - Dataset operations

    fileprivate func h5Dcreate2(parent: hid_t, name: String, datatype: hid_t, dataspace: hid_t) throws -> hid_t {
        let datasetId = name.withCString {
            H5Dcreate2(
                parent,
                $0,
                datatype,
                dataspace,
                hdf5_get_p_default(),
                hdf5_get_p_default(),
                hdf5_get_p_default()
            )
        }
        guard datasetId >= 0 else { throw HDF5Error.datasetCreateFailed(name) }
        return datasetId
    }
    
    fileprivate func h5Dopen2(parent: hid_t, name: String) throws -> hid_t {
        let datasetId = name.withCString {
            H5Dopen2(parent, $0, hdf5_get_p_default())
        }
        guard datasetId >= 0 else { throw HDF5Error.datasetOpenFailed(name) }
        return datasetId
    }
    
    fileprivate func h5Dclose(_ id: hid_t) throws {
        guard H5Dclose(id) >= 0 else {
            throw HDF5Error.datasetCloseFailed
        }
    }
    
    fileprivate func h5Dget_space(dataset: hid_t) throws -> hid_t {
        let spaceId = H5Dget_space(dataset)
        guard spaceId >= 0 else { throw HDF5Error.operationFailed("Failed to get dataspace") }
        return spaceId
    }

    fileprivate func h5Dwrite<T: Sendable>(dataset: hid_t, data: [T]) throws {
        let typeId = H5Dget_type(dataset)
        guard typeId >= 0 else { throw HDF5Error.invalidDataType }
        defer { H5Tclose(typeId) }

        let res = data.withUnsafeBufferPointer { ptr in
            H5Dwrite(
                dataset,
                typeId,
                hdf5_get_s_all(),
                hdf5_get_s_all(),
                hdf5_get_p_default(),
                ptr.baseAddress
            )
        }
        guard res >= 0 else { throw HDF5Error.datasetWriteFailed("") }
    }

    public func readDataset<T: Numeric & Sendable>(_ dataset: HDF5DatasetRef) throws -> [T] {
        let spaceId = H5Dget_space(dataset.id)
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

        let typeId = H5Dget_type(dataset.id)
        guard typeId >= 0 else { throw HDF5Error.invalidDataType }
        defer { H5Tclose(typeId) }

        let res = buffer.withUnsafeMutableBufferPointer { ptr in
            H5Dread(
                dataset.id,
                typeId,
                hdf5_get_s_all(),
                hdf5_get_s_all(),
                hdf5_get_p_default(),
                ptr.baseAddress
            )
        }
        guard res >= 0 else { throw HDF5Error.datasetReadFailed(dataset.name) }
        return buffer
    }

    public func readDataset<T: Sendable>(
        _ dataset: HDF5DatasetRef,
        into buffer: inout [T]
    ) throws {
        let typeId = H5Dget_type(dataset.id)
        guard typeId >= 0 else { throw HDF5Error.invalidDataType }
        defer { H5Tclose(typeId) }

        let res = buffer.withUnsafeMutableBufferPointer { ptr in
            H5Dread(
                dataset.id,
                typeId,
                hdf5_get_s_all(),
                hdf5_get_s_all(),
                hdf5_get_p_default(),
                ptr.baseAddress
            )
        }
        guard res >= 0 else { throw HDF5Error.datasetReadFailed(dataset.name) }
    }

    // MARK: - Attribute operations

    public func writeAttribute<T: Sendable>(
        _ name: String,
        on object: some HDF5Identifiable,
        value: T,
        datatype: hid_t
    ) throws {
        let dataspaceId = H5Screate(hdf5_get_s_scalar())
        guard dataspaceId >= 0 else { throw HDF5Error.dataspaceCreateFailed }
        defer { H5Sclose(dataspaceId) }

        let attrId = name.withCString {
            H5Acreate2(
                object.id,
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

    public func readAttribute<T: Sendable>(_ name: String, from object: some HDF5Identifiable) throws -> T {
        let attrId = name.withCString {
            H5Aopen(object.id, $0, hdf5_get_p_default())
        }
        guard attrId >= 0 else { throw HDF5Error.attributeOpenFailed(name) }
        defer { H5Aclose(attrId) }

        let typeId = H5Aget_type(attrId)
        guard typeId >= 0 else { throw HDF5Error.invalidDataType }
        defer { H5Tclose(typeId) }

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

        return try withUnsafeTemporaryAllocation(of: T.self, capacity: 1) { buffer -> T in
            guard H5Aread(attrId, typeId, buffer.baseAddress) >= 0 else {
                throw HDF5Error.attributeReadFailed(name)
            }
            return buffer[0]
        }
    }
}

// MARK: - Protocols for parent containers

public protocol HDF5Identifiable: Sendable {
    var id: hid_t { get }
}

public protocol HDF5Parent: HDF5Identifiable {
    var parentId: hid_t { get }
    var parentName: String { get }
    var parentFileId: hid_t { get }
}

extension HDF5Parent {
    public func createGroup(_ name: String) async throws -> HDF5GroupRef {
        let groupId = try await HDF5.shared.h5Gcreate2(name, self.id)
        let fullName = self.parentName.isEmpty ? name : "\(self.parentName)/\(name)"
        return HDF5GroupRef(name: fullName, id: groupId, fileId: parentFileId, parent: self)
    }
    
    public func openGroup(_ name: String) async throws -> HDF5GroupRef {
        let groupId = try await HDF5.shared.h5Gopen2(name, self.id)
        let fullName = parentName.isEmpty ? name : "\(parentName)/\(name)"
        return HDF5GroupRef(name: fullName, id: groupId, fileId: parentFileId, parent: self)
    }
    
    public func createDataset(
        _ name: String,
        datatype: hid_t,
        dataspace: HDF5DataspaceRef
    ) async throws -> HDF5DatasetRef {
        let datasetId = try await HDF5.shared.h5Dcreate2(parent: self.id, name: name, datatype: datatype, dataspace: dataspace.id)
        guard datasetId >= 0 else { throw HDF5Error.datasetCreateFailed(name) }
        let fullName = parentName.isEmpty ? name : "\(parentName)/\(name)"
        return HDF5DatasetRef(name: fullName, id: datasetId)
    }
    
    public func openDataset(_ name: String) async throws -> HDF5DatasetRef {
        let datasetId = try await HDF5.shared.h5Dopen2(parent: id, name: name)
        let fullName = parentName.isEmpty ? name : "\(parentName)/\(name)"
        return HDF5DatasetRef(name: fullName, id: datasetId)
    }
}

extension HDF5FileRef: HDF5Identifiable, HDF5Parent {
    public var parentId: hid_t { self.id }
    public var parentName: String { "" }
    public var parentFileId: hid_t { self.id }
}

extension HDF5GroupRef: HDF5Identifiable, HDF5Parent {
    public var parentId: hid_t { self.id }
    public var parentName: String { self.name }
    public var parentFileId: hid_t { self.fileId }
}

extension HDF5DatasetRef: HDF5Identifiable {}
extension HDF5DataspaceRef: HDF5Identifiable {}
