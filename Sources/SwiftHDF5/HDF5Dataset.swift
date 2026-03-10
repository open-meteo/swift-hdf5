/// A handle to an open HDF5 dataset within a file.
///
/// Datasets are the primary storage containers in an HDF5 file. They hold
/// a multi-dimensional array of elements all sharing the same datatype, whose
/// shape is described by an ``HDF5Dataspace``.
///
/// Obtain an instance via ``HDF5FileOrGroup/createDataset(_:datatype:dataspace:)``
/// or ``HDF5FileOrGroup/openDataset(_:)``. The dataset is closed automatically
/// when the last reference to this object is released.
///
/// `HDF5Dataset` conforms to ``HDF5Attributable``, so you can attach scalar
/// metadata attributes (units, scale factors, etc.) directly to the dataset.
///
/// ```swift
/// let space   = try await HDF5.createDataspace(dimensions: [5])
/// let dataset = try await file.createDataset(
///     "temperatures",
///     datatype: HDF5Datatype.double,
///     dataspace: space
/// )
/// try await dataset.writeDataset(data: [20.5, 21.3, 19.8, 22.1, 20.9])
/// try await dataset.writeAttribute("unit", value: "Celsius")
/// ```
public final class HDF5Dataset: Sendable {
    let id: hid_t
    let parent: any HDF5FileOrGroupImpl

    init(id: hid_t, parent: any HDF5FileOrGroupImpl) {
        self.id = id
        self.parent = parent
    }

    /// Writes `data` into this dataset, replacing any previously stored values.
    ///
    /// The element type `T` must match the HDF5 datatype that was specified when
    /// the dataset was created (e.g. `Int32` for ``HDF5Datatype/int32``). The
    /// number of elements in `data` must equal the total number of elements
    /// described by the dataset's dataspace (the product of all dimensions).
    ///
    /// - Parameter data: A flat array of values to write. For multi-dimensional
    ///   datasets the array is in row-major (C) order.
    /// - Throws: ``HDF5Error/datasetWriteFailed(_:)`` if the C library returns an
    ///   error, or ``HDF5Error/invalidDataType`` if the stored type identifier
    ///   cannot be retrieved.
    public func writeDataset<T: HDF5DatasetType>(data: [T]) async throws {
        try await HDF5.h5Dwrite(dataset: id, data: data)
    }

    /// Reads all elements of this dataset and returns them as a flat array.
    ///
    /// The element type `T` is inferred from the call-site type annotation and
    /// must be compatible with the datatype stored in the file. A type mismatch
    /// is detected at runtime and throws ``HDF5Error/datasetTypeMismatch(expected:actual:)``.
    ///
    /// For multi-dimensional datasets the elements are returned in row-major
    /// (C) order. Use ``space`` to retrieve the dimensions if you need to
    /// reshape the result.
    ///
    /// - Returns: A newly allocated array containing all dataset elements.
    /// - Throws: ``HDF5Error/datasetReadFailed(_:)`` if the C library returns an
    ///   error, ``HDF5Error/datasetTypeMismatch(expected:actual:)`` if `T` does
    ///   not match the stored type, or ``HDF5Error/invalidDataType`` if the type
    ///   identifier cannot be retrieved.
    public func readDataset<T: HDF5DatasetType>() async throws -> [T] { return try await HDF5.readDataset(id) }

    /// Reads all elements of this dataset into a pre-allocated buffer and
    /// returns it, avoiding an extra allocation compared to ``readDataset()``.
    ///
    /// `buffer` must have exactly as many elements as the dataset (the product
    /// of all dimension sizes). The same type-compatibility rules apply as for
    /// ``readDataset()``.
    ///
    /// - Parameter buffer: A consuming array whose storage is reused for the
    ///   read result. Its count must match the total number of dataset elements.
    /// - Returns: The same buffer, now filled with the dataset's values.
    /// - Throws: ``HDF5Error/datasetReadFailed(_:)`` if the C library returns an
    ///   error, ``HDF5Error/datasetTypeMismatch(expected:actual:)`` if `T` does
    ///   not match the stored type, or ``HDF5Error/invalidDataType`` if the type
    ///   identifier cannot be retrieved.
    public func readDataset<T: HDF5DatasetType>(reusing buffer: consuming [T]) async throws -> [T] {
        try await HDF5.readDataset(id, reusing: consume buffer)
    }

    /// The dataspace that describes the shape of this dataset.
    ///
    /// Use the returned ``HDF5Dataspace/dimensions`` property to retrieve the
    /// size of each dimension.
    ///
    /// ```swift
    /// let dims = try await dataset.space.dimensions
    /// // e.g. [3, 4] for a 3×4 matrix
    /// ```
    ///
    /// - Throws: ``HDF5Error/operationFailed(_:)`` if the dataspace cannot be
    ///   retrieved from the C library.
    public var space: HDF5Dataspace {
        get async throws {
            let spaceId = try await HDF5.h5Dget_space(dataset: id)
            return HDF5Dataspace(id: spaceId)
        }
    }

    deinit { try? HDF5.h5Dclose(id) }
}
