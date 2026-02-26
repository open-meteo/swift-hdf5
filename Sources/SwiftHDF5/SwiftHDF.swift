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

public struct HDF5FileRef: Sendable {
    public let path: String
    public let id: hid_t
}

public struct HDF5GroupRef: Sendable {
    public let name: String
    public let id: hid_t
    public let fileId: hid_t
}

public struct HDF5DatasetRef: Sendable {
    public let name: String
    public let id: hid_t
}

public struct HDF5DataspaceRef: Sendable {
    public let name: String
    public let id: hid_t
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
        hdf5_disable_auto_print()
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

    public func closeFile(_ file: consuming HDF5FileRef) throws {
        guard H5Fclose(file.id) >= 0 else {
            throw HDF5Error.operationFailed("Failed to close file: \(file.path)")
        }
    }

    // MARK: - Group operations

    public func createGroup(_ name: String, in parent: some HDF5Parent) throws -> HDF5GroupRef {
        let groupId = name.withCString {
            H5Gcreate2(
                parent.parentId,
                $0,
                hdf5_get_p_default(),
                hdf5_get_p_default(),
                hdf5_get_p_default()
            )
        }
        guard groupId >= 0 else { throw HDF5Error.groupCreateFailed(name) }
        let fullName = parent.parentName.isEmpty ? name : "\(parent.parentName)/\(name)"
        return HDF5GroupRef(name: fullName, id: groupId, fileId: parent.parentFileId)
    }

    public func openGroup(_ name: String, in parent: some HDF5Parent) throws -> HDF5GroupRef {
        let groupId = name.withCString {
            H5Gopen2(parent.parentId, $0, hdf5_get_p_default())
        }
        guard groupId >= 0 else { throw HDF5Error.groupOpenFailed(name) }
        let fullName = parent.parentName.isEmpty ? name : "\(parent.parentName)/\(name)"
        return HDF5GroupRef(name: fullName, id: groupId, fileId: parent.parentFileId)
    }

    public func closeGroup(_ group: consuming HDF5GroupRef) throws {
        guard H5Gclose(group.id) >= 0 else {
            throw HDF5Error.operationFailed("Failed to close group: \(group.name)")
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

    public func closeDataspace(_ space: HDF5DataspaceRef) {
        H5Sclose(space.id)
    }

    public func getDimensions(_ space: HDF5DataspaceRef) throws -> [hsize_t] {
        let ndims = H5Sget_simple_extent_ndims(space.id)
        guard ndims >= 0 else { throw HDF5Error.operationFailed("Failed to get dimensions") }

        var dims = [hsize_t](repeating: 0, count: Int(ndims))
        let res = dims.withUnsafeMutableBufferPointer { ptr in
            H5Sget_simple_extent_dims(space.id, ptr.baseAddress, nil)
        }
        guard res >= 0 else { throw HDF5Error.operationFailed("Failed to get dimensions") }
        return dims
    }

    // MARK: - Dataset operations

    public func createDataset(
        _ name: String,
        in parent: some HDF5Parent,
        datatype: hid_t,
        dataspace: HDF5DataspaceRef
    ) throws -> HDF5DatasetRef {
        let datasetId = name.withCString {
            H5Dcreate2(
                parent.parentId,
                $0,
                datatype,
                dataspace.id,
                hdf5_get_p_default(),
                hdf5_get_p_default(),
                hdf5_get_p_default()
            )
        }
        guard datasetId >= 0 else { throw HDF5Error.datasetCreateFailed(name) }
        let fullName = parent.parentName.isEmpty ? name : "\(parent.parentName)/\(name)"
        return HDF5DatasetRef(name: fullName, id: datasetId)
    }

    public func openDataset(_ name: String, in parent: some HDF5Parent) throws -> HDF5DatasetRef {
        let datasetId = name.withCString {
            H5Dopen2(parent.parentId, $0, hdf5_get_p_default())
        }
        guard datasetId >= 0 else { throw HDF5Error.datasetOpenFailed(name) }
        let fullName = parent.parentName.isEmpty ? name : "\(parent.parentName)/\(name)"
        return HDF5DatasetRef(name: fullName, id: datasetId)
    }

    public func closeDataset(_ dataset: consuming HDF5DatasetRef) throws {
        guard H5Dclose(dataset.id) >= 0 else {
            throw HDF5Error.operationFailed("Failed to close dataset: \(dataset.name)")
        }
    }

    public func getDatasetSpace(_ dataset: HDF5DatasetRef) throws -> HDF5DataspaceRef {
        let spaceId = H5Dget_space(dataset.id)
        guard spaceId >= 0 else { throw HDF5Error.operationFailed("Failed to get dataspace") }
        return HDF5DataspaceRef(name: "", id: spaceId)
    }

    public func writeDataset<T: Sendable>(_ dataset: HDF5DatasetRef, data: [T]) throws {
        let typeId = H5Dget_type(dataset.id)
        guard typeId >= 0 else { throw HDF5Error.invalidDataType }
        defer { H5Tclose(typeId) }

        let res = data.withUnsafeBufferPointer { ptr in
            H5Dwrite(
                dataset.id,
                typeId,
                hdf5_get_s_all(),
                hdf5_get_s_all(),
                hdf5_get_p_default(),
                ptr.baseAddress
            )
        }
        guard res >= 0 else { throw HDF5Error.datasetWriteFailed(dataset.name) }
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
        into buffer: consuming [T]
    ) throws -> [T] {
        var buffer = consume buffer

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

    public func readErrorStack() -> String {
        var messages: [String] = []

        // H5Ewalk2 walks the error stack and calls the closure for each frame
        withUnsafeMutablePointer(to: &messages) { ptr in
            H5Ewalk2(
                hdf5_get_e_default(),
                H5E_WALK_DOWNWARD,
                { _, errorInfo, userData in
                    guard let errorInfo = errorInfo,
                        let userData = userData
                    else { return 0 }

                    let messagesPtr = userData.assumingMemoryBound(to: [String].self)

                    // Read the minor message directly from the error info struct
                    if let desc = errorInfo.pointee.desc {
                        messagesPtr.pointee.append(String(cString: desc))
                    }
                    if let funcName = errorInfo.pointee.func_name {
                        messagesPtr.pointee.append(String(cString: funcName))
                    }

                    return 0
                },
                ptr
            )
        }

        H5Eclear2(hdf5_get_e_default())
        return messages.joined(separator: " | ")
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
