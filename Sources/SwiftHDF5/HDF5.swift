import CHDF5
import Foundation

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

public typealias hid_t = CHDF5.hid_t

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


// MARK: - Serializes ALL HDF5 C calls on a single background thread to ensure thread safety

public enum HDF5 {
    private static let queue = DispatchQueue(
        label: "SwiftHDF5.single-thread"
    )

    private static func execute<T: Sendable>(_ work: @escaping @Sendable () -> sending T) async -> sending T {
        await withCheckedContinuation { continuation in
            queue.async {
                let result = work()
                continuation.resume(returning: result)
            }
        }
    }
    
    static func execute<T: Sendable>(_ work: @escaping @Sendable () throws -> sending T) async throws -> sending T {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    let result = try work()
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    // MARK: - File operations

    static public func createFile(_ path: String, mode: FileAccessMode = .truncate) async throws -> HDF5File {
        let fileId = await execute {
            path.withCString { cPath in
                H5Fcreate(cPath, mode.cMode, hdf5_get_p_default(), hdf5_get_p_default())
            }
        }
        guard fileId >= 0 else { throw HDF5Error.fileCreateFailed(path) }
        return HDF5File(id: fileId)
    }

    static public func openFile(_ path: String, mode: FileAccessMode = .readOnly) async throws -> HDF5File {
        let fileId = await execute {
            path.withCString { cPath in
                H5Fopen(cPath, mode.cMode, hdf5_get_p_default())
            }
        }
        guard fileId >= 0 else { throw HDF5Error.fileOpenFailed(path) }
        return HDF5File(id: fileId)
    }
    
    static func h5Fclose(_ id: hid_t) throws {
        guard queue.sync(execute: { H5Fclose(id) }) >= 0 else {
            throw HDF5Error.fileCloseFailed
        }
    }

    // MARK: - Group operations
    
    static func h5Gcreate2(_ name: String, _ parentId: hid_t) async throws -> hid_t {
        let groupId = await execute {
            name.withCString {
                H5Gcreate2(
                    parentId,
                    $0,
                    hdf5_get_p_default(),
                    hdf5_get_p_default(),
                    hdf5_get_p_default()
                )
            }
        }
        guard groupId >= 0 else { throw HDF5Error.groupCreateFailed(name) }
        return groupId
    }
    
    static func h5Gopen2(_ name: String, _ parentId: hid_t) async throws -> hid_t {
        let groupId = await execute {
            name.withCString {
                H5Gopen2(parentId, $0, hdf5_get_p_default())
            }
        }
        guard groupId >= 0 else { throw HDF5Error.groupOpenFailed(name) }
        return groupId
    }
    
    static func h5Gclose(_ id: hid_t) throws {
        guard queue.sync(execute: { H5Gclose(id) }) >= 0 else {
            throw HDF5Error.groupCloseFailed
        }
    }

    // MARK: - Dataspace operations

    static public func createDataspace(dimensions: [hsize_t]) async throws -> HDF5Dataspace {
        let spaceId = await execute {
            dimensions.withUnsafeBufferPointer { ptr in
                H5Screate_simple(Int32(dimensions.count), ptr.baseAddress, nil)
            }
        }
        guard spaceId >= 0 else { throw HDF5Error.dataspaceCreateFailed }
        return HDF5Dataspace(id: spaceId)
    }
    
//    fileprivate func h5Screate_simple(dimensions: [hsize_t]) -> hid_t {
//        return dimensions.withUnsafeBufferPointer { ptr in
//            H5Screate_simple(Int32(dimensions.count), ptr.baseAddress, nil)
//        }
//    }
    
    static func h5Sget_simple_extent_dims(space_id: hid_t) async throws -> [hsize_t]  {
        return try await execute {
            let ndims = H5Sget_simple_extent_ndims(space_id)
            guard ndims >= 0 else { throw HDF5Error.operationFailed("Failed to get dimensions") }
            
            var dims = [hsize_t](repeating: 0, count: Int(ndims))
            let res = dims.withUnsafeMutableBufferPointer { ptr in
                H5Sget_simple_extent_dims(space_id, ptr.baseAddress, nil)
            }
            guard res >= 0 else { throw HDF5Error.operationFailed("Failed to get dimensions") }
            return dims
        }
    }
    
    static func h5Sclose(_ id: hid_t) throws {
        guard queue.sync(execute: { H5Sclose(id) }) >= 0 else {
            throw HDF5Error.dataspaceCloseFailed
        }
    }

    // MARK: - Dataset operations

    static func h5Dcreate2(parent: hid_t, name: String, datatype: hid_t, dataspace: hid_t) async throws -> hid_t {
        let datasetId = await execute {
            name.withCString {
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
        }
        guard datasetId >= 0 else { throw HDF5Error.datasetCreateFailed(name) }
        return datasetId
    }
    
    static func h5Dopen2(parent: hid_t, name: String) async throws -> hid_t {
        let datasetId = await execute {
            name.withCString {
                H5Dopen2(parent, $0, hdf5_get_p_default())
            }
        }
        guard datasetId >= 0 else { throw HDF5Error.datasetOpenFailed(name) }
        return datasetId
    }
    
    static func h5Dclose(_ id: hid_t) throws {
        guard queue.sync(execute: { H5Dclose(id) }) >= 0 else {
            throw HDF5Error.datasetCloseFailed
        }
    }
    
    static func h5Dget_space(dataset: hid_t) async throws -> hid_t {
        let spaceId = await execute { H5Dget_space(dataset) }
        guard spaceId >= 0 else { throw HDF5Error.operationFailed("Failed to get dataspace") }
        return spaceId
    }

    static func h5Dwrite<T: Sendable>(dataset: hid_t, data: [T]) async throws {
        return try await execute {
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
    }

    static func readDataset<T: Numeric & Sendable>(_ dataset: hid_t) async throws -> [T] {
        return try await execute {
            let spaceId = H5Dget_space(dataset)
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
            
            let typeId = H5Dget_type(dataset)
            guard typeId >= 0 else { throw HDF5Error.invalidDataType }
            defer { H5Tclose(typeId) }
            
            let res = buffer.withUnsafeMutableBufferPointer { ptr in
                H5Dread(
                    dataset,
                    typeId,
                    hdf5_get_s_all(),
                    hdf5_get_s_all(),
                    hdf5_get_p_default(),
                    ptr.baseAddress
                )
            }
            guard res >= 0 else { throw HDF5Error.datasetReadFailed("dataset.name") }
            return buffer
        }
    }
//
//    static func readDataset<T: Sendable>(
//        _ dataset: hid_t,
//        into buffer: inout [T]
//    ) async throws {
//        return try await execute {
//            let typeId = H5Dget_type(dataset)
//            guard typeId >= 0 else { throw HDF5Error.invalidDataType }
//            defer { H5Tclose(typeId) }
//            
//            let res = buffer.withUnsafeMutableBufferPointer { ptr in
//                H5Dread(
//                    dataset,
//                    typeId,
//                    hdf5_get_s_all(),
//                    hdf5_get_s_all(),
//                    hdf5_get_p_default(),
//                    ptr.baseAddress
//                )
//            }
//            guard res >= 0 else { throw HDF5Error.datasetReadFailed("dataset.name") }
//        }
//    }

    // MARK: - Attribute operations

    static func writeAttribute<T: Sendable>(
        _ name: String,
        on object: hid_t,
        value: T,
        datatype: hid_t
    ) async throws {
        try await execute {
            let dataspaceId = H5Screate(hdf5_get_s_scalar())
            guard dataspaceId >= 0 else { throw HDF5Error.dataspaceCreateFailed }
            defer { H5Sclose(dataspaceId) }
            
            let attrId = name.withCString {
                H5Acreate2(
                    object,
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

    static func readAttribute<T: Sendable>(_ name: String, from object: hid_t) async throws -> T {
        return try await execute {
            let attrId = name.withCString {
                H5Aopen(object, $0, hdf5_get_p_default())
            }
            guard attrId >= 0 else { throw HDF5Error.attributeOpenFailed(name) }
            defer { H5Aclose(attrId) }
            
            let typeId = H5Aget_type(attrId)
            guard typeId >= 0 else { throw HDF5Error.invalidDataType }
            defer { H5Tclose(typeId) }
            
            if T.self == String.self {
                let size = H5Tget_size(typeId)
                guard size > 0 else { return "" as! T }
                // TODO: use String(unsafeUninitializedCapacity
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
    
    static func h5Iget_name(id: hid_t) async throws -> String {
        return try await execute {
            let size = H5Iget_name(id, nil, 0)
            guard size >= 0 else { throw HDF5Error.operationFailed("Failed to get name size") }
            // Capacity need to be +1 to accommodate the null terminator
            let name = try String(unsafeUninitializedCapacity: size+1, initializingUTF8With: { ptr in
                let result = H5Iget_name(id, ptr.baseAddress, ptr.count)
                guard result >= 0 else { throw HDF5Error.operationFailed("Failed to get name") }
                guard result == size else { throw HDF5Error.operationFailed("Failed to get name") }
                return size
            })
            return name
        }
    }
    
    static func h5Fget_name(id: hid_t) async throws -> String {
        return try await execute {
            let size = H5Fget_name(id, nil, 0)
            guard size >= 0 else { throw HDF5Error.operationFailed("Failed to get file name size") }
            // Capacity need to be +1 to accommodate the null terminator
            let name = try String(unsafeUninitializedCapacity: size+1, initializingUTF8With: { ptr in
                let result = H5Fget_name(id, ptr.baseAddress, ptr.count)
                guard result >= 0 else { throw HDF5Error.operationFailed("Failed to get file name") }
                guard result == size else { throw HDF5Error.operationFailed("Failed to get file name") }
                return size
            })
            return name
        }
    }
}

