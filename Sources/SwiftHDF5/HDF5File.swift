/// A handle to an open HDF5 file.
///
/// Obtain an instance via ``HDF5/createFile(_:mode:)`` or ``HDF5/openFile(_:mode:)``.
/// The file is closed automatically when the last reference to this object is released.
///
/// `HDF5File` conforms to ``HDF5FileOrGroup``, so you can create and open groups
/// and datasets directly on it, just as you would on an ``HDF5Group``.
///
/// ```swift
/// // Create a file, write a dataset, then let the handle close automatically.
/// let file = try await HDF5.createFile("/tmp/data.h5")
/// let space = try await HDF5.createDataspace(dimensions: [10])
/// let ds    = try await file.createDataset("values", datatype: HDF5Datatype.int32, dataspace: space)
/// try await ds.writeDataset(data: [Int32](0..<10))
/// ```
public final class HDF5File: Sendable {
    let id: hid_t

    init(id: hid_t) {
        self.id = id
    }

    deinit {
        try? HDF5.h5Fclose(id)
    }
}
