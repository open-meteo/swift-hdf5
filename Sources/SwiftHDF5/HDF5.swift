import CHDF5
import Foundation

/// Controls how an HDF5 file is opened or created.
///
/// Pass one of these values to ``HDF5/createFile(_:mode:)`` or
/// ``HDF5/openFile(_:mode:)`` to specify the desired access behaviour.
public enum FileAccessMode: Sendable {
    /// Open an existing file for reading only. The file must already exist;
    /// if it does not, ``HDF5Error/fileOpenFailed(_:)`` is thrown.
    case readOnly
    /// Open an existing file for both reading and writing. The file must
    /// already exist; if it does not, ``HDF5Error/fileOpenFailed(_:)`` is thrown.
    case readWrite
    /// Create a new file, overwriting it if it already exists. This is the
    /// default mode for ``HDF5/createFile(_:mode:)``.
    case truncate
    /// Create a new file, failing with ``HDF5Error/fileCreateFailed(_:)`` if a
    /// file at the given path already exists.
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

// swift-format-ignore: TypeNamesShouldBeCapitalized
public typealias hid_t = CHDF5.hid_t

// MARK: - HDF5 Datatype

/// Native HDF5 type identifiers for use with ``HDF5FileOrGroup/createDataset(_:datatype:dataspace:)``.
///
/// Each property returns the HDF5 identifier (`hid_t`) for the corresponding
/// platform-native C type. The mapping to Swift types is:
///
/// | Property       | Swift type  |
/// |----------------|-------------|
/// | ``int8``       | `Int8`      |
/// | ``int16``      | `Int16`     |
/// | ``int32``      | `Int32`     |
/// | ``int64``      | `Int64`     |
/// | ``uint8``      | `UInt8`     |
/// | ``uint16``     | `UInt16`    |
/// | ``uint32``     | `UInt32`    |
/// | ``uint64``     | `UInt64`    |
/// | ``float``      | `Float`     |
/// | ``double``     | `Double`    |
/// | ``char``       | `CChar`     |
public enum HDF5Datatype {
    /// The native signed 8-bit integer type (`H5T_NATIVE_INT8`). Corresponds to Swift's `Int8`.
    public static var int8: hid_t { hdf5_get_native_int8() }
    /// The native signed 16-bit integer type (`H5T_NATIVE_INT16`). Corresponds to Swift's `Int16`.
    public static var int16: hid_t { hdf5_get_native_int16() }
    /// The native signed 32-bit integer type (`H5T_NATIVE_INT32`). Corresponds to Swift's `Int32`.
    public static var int32: hid_t { hdf5_get_native_int32() }
    /// The native signed 64-bit integer type (`H5T_NATIVE_INT64`). Corresponds to Swift's `Int64`.
    public static var int64: hid_t { hdf5_get_native_int64() }
    /// The native unsigned 8-bit integer type (`H5T_NATIVE_UINT8`). Corresponds to Swift's `UInt8`.
    public static var uint8: hid_t { hdf5_get_native_uint8() }
    /// The native unsigned 16-bit integer type (`H5T_NATIVE_UINT16`). Corresponds to Swift's `UInt16`.
    public static var uint16: hid_t { hdf5_get_native_uint16() }
    /// The native unsigned 32-bit integer type (`H5T_NATIVE_UINT32`). Corresponds to Swift's `UInt32`.
    public static var uint32: hid_t { hdf5_get_native_uint32() }
    /// The native unsigned 64-bit integer type (`H5T_NATIVE_UINT64`). Corresponds to Swift's `UInt64`.
    public static var uint64: hid_t { hdf5_get_native_uint64() }
    /// The native 32-bit floating-point type (`H5T_NATIVE_FLOAT`). Corresponds to Swift's `Float`.
    public static var float: hid_t { hdf5_get_native_float() }
    /// The native 64-bit floating-point type (`H5T_NATIVE_DOUBLE`). Corresponds to Swift's `Double`.
    public static var double: hid_t { hdf5_get_native_double() }
    /// The native C `char` type (`H5T_NATIVE_CHAR`). Corresponds to Swift's `CChar`.
    public static var char: hid_t { hdf5_get_native_char() }
}

// MARK: - Thread-safe entry point

/// The top-level namespace for creating and opening HDF5 files and dataspaces.
///
/// ## Thread safety
///
/// The HDF5 C library is **not thread-safe by default**. All calls into the C
/// library are serialised through a single internal `DispatchQueue`
/// (`SwiftHDF5.single-thread`). Every `async` method on this type and on the
/// objects it returns (`HDF5File`, `HDF5Group`, `HDF5Dataset`, …) suspends the
/// caller and resumes on that queue, so concurrent calls from multiple Swift
/// `Task`s are safe — they are simply queued and executed one at a time.
///
/// There is no need to add any additional synchronisation when using this
/// library from multiple tasks or actors.
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

    private static func execute<T: Sendable>(
        _ work: @escaping @Sendable () throws -> sending T
    ) async throws -> sending T {
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

    /// Creates a new HDF5 file at `path`.
    ///
    /// - Parameters:
    ///   - path: The filesystem path at which the file should be created.
    ///   - mode: How the file should be created. Defaults to ``FileAccessMode/truncate``,
    ///     which overwrites any existing file. Use ``FileAccessMode/exclusive`` to
    ///     fail instead of overwriting.
    /// - Returns: An ``HDF5File`` handle for the newly created file.
    /// - Throws: ``HDF5Error/fileCreateFailed(_:)`` if the C library returns an error
    ///   (e.g. the parent directory does not exist, or the file already exists when
    ///   using ``FileAccessMode/exclusive``).
    static public func createFile(_ path: String, mode: FileAccessMode = .truncate) async throws -> HDF5File {
        let fileId = await execute {
            path.withCString { cPath in
                H5Fcreate(cPath, mode.cMode, hdf5_get_p_default(), hdf5_get_p_default())
            }
        }
        guard fileId >= 0 else { throw HDF5Error.fileCreateFailed(path) }
        return HDF5File(id: fileId)
    }

    /// Opens an existing HDF5 file at `path`.
    ///
    /// - Parameters:
    ///   - path: The filesystem path of the file to open.
    ///   - mode: The access mode. Defaults to ``FileAccessMode/readOnly``. Use
    ///     ``FileAccessMode/readWrite`` to open the file for writing without
    ///     truncating it.
    /// - Returns: An ``HDF5File`` handle for the opened file.
    /// - Throws: ``HDF5Error/fileOpenFailed(_:)`` if the file does not exist or
    ///   cannot be opened with the requested access mode.
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

    /// Creates a simple (rectilinear) dataspace with the given dimensions.
    ///
    /// Pass the resulting ``HDF5Dataspace`` to
    /// ``HDF5FileOrGroup/createDataset(_:datatype:dataspace:)`` to define the
    /// shape of a new dataset.
    ///
    /// - Parameter dimensions: The size of each dimension, in elements. For
    ///   example, `[3, 4]` produces a 3×4 matrix dataspace.
    /// - Returns: An ``HDF5Dataspace`` representing the described shape.
    /// - Throws: ``HDF5Error/dataspaceCreateFailed`` if the C library returns
    ///   an error (e.g. `dimensions` is empty).
    static public func createDataspace(dimensions: [hsize_t]) async throws -> HDF5Dataspace {
        let spaceId = await execute {
            dimensions.withUnsafeBufferPointer { ptr in
                H5Screate_simple(Int32(dimensions.count), ptr.baseAddress, nil)
            }
        }
        guard spaceId >= 0 else { throw HDF5Error.dataspaceCreateFailed }
        return HDF5Dataspace(id: spaceId)
    }

    static func h5Sget_simple_extent_dims(space_id: hid_t) async throws -> [hsize_t] {
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

    static func h5Dwrite<T: HDF5DatasetType>(dataset: hid_t, data: [T]) async throws {
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
            guard res >= 0 else { throw HDF5Error.datasetWriteFailed("id: \(dataset)") }
        }
    }

    static func readDataset<T: HDF5DatasetType>(_ dataset: hid_t) async throws -> [T] {
        let buffer = try await execute {
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
            return [T](repeating: T.defaultValue, count: Int(count))
        }
        return try await self.readDataset(dataset, reusing: buffer)
    }

    static func readDataset<T: HDF5DatasetType>(
        _ dataset: hid_t,
        reusing buffer: consuming [T]
    ) async throws -> [T] {
        nonisolated(unsafe) var buffer = consume buffer
        return try await execute {

            let typeId = H5Dget_type(dataset)
            guard typeId >= 0 else { throw HDF5Error.invalidDataType }
            defer { H5Tclose(typeId) }

            // Validate type class and byte width against T before reading.
            // H5Tequal is not used because it rejects equivalent types that differ
            // only in byte order or native-vs-IEEE encoding (e.g. H5T_NATIVE_FLOAT
            // vs H5T_IEEE_F32LE). Comparing class + size is the portable alternative
            // and matches the approach used by h5py and the HDF5 C++ API.
            let storedClass = H5Tget_class(typeId)
            let storedSize = H5Tget_size(typeId)
            guard storedClass == T.hdf5TypeClass && storedSize == T.hdf5TypeSize else {
                throw HDF5Error.datasetTypeMismatch(
                    expected: "\(T.self) (\(T.hdf5TypeClass.rawValue)/\(T.hdf5TypeSize)B)",
                    actual: T.hdf5TypeDescription(typeId)
                )
            }

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
            guard res >= 0 else { throw HDF5Error.datasetReadFailed("id: \(dataset)") }

            return buffer
        }
    }

    // MARK: - Attribute operations

    /// Write an attribute named `name` on the given HDF5 object using the HDF5
    /// datatype associated with `T` (via `T.hdf5TypeId`). The datatype is
    /// inferred from the Swift type, so callers do not need to supply it.
    static func writeAttribute<T: HDF5AttributeType>(
        _ name: String,
        on object: hid_t,
        value: T
    ) async throws {
        if T.self == String.self {
            // Strings need their own write path: the datatype owns heap memory and
            // must be closed after use, and H5Awrite expects a pointer-to-pointer.
            try await writeStringAttribute(name: name, on: object, value: value as! String)
        } else {
            try await execute {
                let dataspaceId = H5Screate(hdf5_get_s_scalar())
                guard dataspaceId >= 0 else { throw HDF5Error.dataspaceCreateFailed }
                defer { H5Sclose(dataspaceId) }

                let attrId = name.withCString {
                    H5Acreate2(
                        object,
                        $0,
                        T.hdf5TypeId,
                        dataspaceId,
                        hdf5_get_p_default(),
                        hdf5_get_p_default()
                    )
                }
                guard attrId >= 0 else { throw HDF5Error.attributeCreateFailed(name) }
                defer { H5Aclose(attrId) }

                var mutableValue = value
                let res = withUnsafePointer(to: &mutableValue) { ptr in
                    H5Awrite(attrId, T.hdf5TypeId, ptr)
                }
                guard res >= 0 else { throw HDF5Error.attributeWriteFailed(name) }
            }
        }
    }

    /// Helper that performs the string-specific attribute creation and write.
    private static func writeStringAttribute(name: String, on object: hid_t, value: String) async throws {
        try await execute {
            let typeId = String.hdf5TypeId
            guard typeId >= 0 else { throw HDF5Error.invalidDataType }
            defer { H5Tclose(typeId) }

            let dataspaceId = H5Screate(hdf5_get_s_scalar())
            guard dataspaceId >= 0 else { throw HDF5Error.dataspaceCreateFailed }
            defer { H5Sclose(dataspaceId) }

            let attrId = name.withCString {
                H5Acreate2(
                    object,
                    $0,
                    typeId,
                    dataspaceId,
                    hdf5_get_p_default(),
                    hdf5_get_p_default()
                )
            }
            guard attrId >= 0 else { throw HDF5Error.attributeCreateFailed(name) }
            defer { H5Aclose(attrId) }

            // H5Awrite for variable-length strings expects a `const char **`.
            let str = value
            let res = str.withCString { cStr -> herr_t in
                var ptr: UnsafePointer<CChar>? = cStr
                return withUnsafePointer(to: &ptr) { H5Awrite(attrId, typeId, $0) }
            }
            guard res >= 0 else { throw HDF5Error.attributeWriteFailed(name) }
        }
    }

    static func readAttribute<T: HDF5AttributeType>(_ name: String, from object: hid_t) async throws -> T {
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
                // Distinguish variable-length strings (H5T_VARIABLE) from fixed-length ones.
                // For variable-length types H5Aread fills a `char *` allocated by HDF5 that
                // must be reclaimed via H5Treclaim; H5Tget_size returns sizeof(char *), not
                // the string length, so it cannot be used to size the read buffer.
                let isVlen = H5Tis_variable_str(typeId) > 0
                let str: String
                if isVlen {
                    // H5Aread expects a pointer to a `char *` (i.e. `char **`).
                    var cStr: UnsafeMutablePointer<CChar>? = nil
                    let dataspaceId = H5Aget_space(attrId)
                    defer { if dataspaceId >= 0 { H5Sclose(dataspaceId) } }
                    let res = withUnsafeMutablePointer(to: &cStr) { H5Aread(attrId, typeId, $0) }
                    guard res >= 0 else { throw HDF5Error.attributeReadFailed(name) }
                    // Copy into Swift string before reclaiming the HDF5 allocation.
                    str = cStr.map { String(cString: $0) } ?? ""
                    // Release memory allocated by HDF5 for the vlen string.
                    _ = withUnsafeMutablePointer(to: &cStr) { ptr in
                        hdf5_vlen_reclaim(typeId, dataspaceId, hdf5_get_p_default(), ptr)
                    }
                } else {
                    let size = H5Tget_size(typeId)
                    guard size > 0 else { return "" as! T }
                    str = try String(unsafeUninitializedCapacity: size + 1) { ptr in
                        let res = H5Aread(attrId, typeId, ptr.baseAddress)
                        guard res >= 0 else { throw HDF5Error.attributeReadFailed(name) }
                        return ptr.firstIndex(of: 0) ?? size
                    }.trimmingCharacters(in: .whitespaces)
                }
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
            let name = try String(
                unsafeUninitializedCapacity: size + 1,
                initializingUTF8With: { ptr in
                    let result = H5Iget_name(id, ptr.baseAddress, ptr.count)
                    guard result >= 0 else { throw HDF5Error.operationFailed("Failed to get name") }
                    guard result == size else { throw HDF5Error.operationFailed("Failed to get name") }
                    return size
                }
            )
            return name
        }
    }

    static func h5Fget_name(id: hid_t) async throws -> String {
        return try await execute {
            let size = H5Fget_name(id, nil, 0)
            guard size >= 0 else { throw HDF5Error.operationFailed("Failed to get file name size") }
            // Capacity need to be +1 to accommodate the null terminator
            let name = try String(
                unsafeUninitializedCapacity: size + 1,
                initializingUTF8With: { ptr in
                    let result = H5Fget_name(id, ptr.baseAddress, ptr.count)
                    guard result >= 0 else { throw HDF5Error.operationFailed("Failed to get file name") }
                    guard result == size else { throw HDF5Error.operationFailed("Failed to get file name") }
                    return size
                }
            )
            return name
        }
    }
}
