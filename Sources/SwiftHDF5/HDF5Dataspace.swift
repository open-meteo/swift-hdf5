/// A handle to an HDF5 dataspace, which describes the shape of a dataset.
///
/// A dataspace defines the number of dimensions and the size of each dimension
/// for a dataset. Create one with ``HDF5/createDataspace(dimensions:)`` before
/// calling ``HDF5FileOrGroup/createDataset(_:datatype:dataspace:)``, or retrieve
/// the dataspace of an existing dataset via ``HDF5Dataset/space``.
///
/// The dataspace is closed automatically when the last reference to this object
/// is released.
///
/// ```swift
/// // Create a 3×4 matrix dataspace
/// let space = try await HDF5.createDataspace(dimensions: [3, 4])
/// let dims  = try await space.dimensions  // → [3, 4]
/// ```
public final class HDF5Dataspace: Sendable {
    let id: hid_t

    init(id: hid_t) {
        self.id = id
    }

    /// The size of each dimension of this dataspace, in elements.
    ///
    /// For a 1-D dataspace of 1024 elements this returns `[1024]`. For a 3×4
    /// matrix it returns `[3, 4]`. The length of the array equals the number
    /// of dimensions (the *rank*) of the dataspace.
    ///
    /// - Throws: ``HDF5Error/operationFailed(_:)`` if the rank or dimension
    ///   sizes cannot be retrieved from the C library.
    public var dimensions: [UInt64] {
        get async throws { return try await HDF5.h5Sget_simple_extent_dims(space_id: id) }
    }

    deinit {
        try? HDF5.h5Sclose(id)
    }
}
