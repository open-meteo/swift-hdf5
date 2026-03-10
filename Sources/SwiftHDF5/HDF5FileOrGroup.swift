/// A container that can hold HDF5 groups and datasets.
///
/// This protocol is adopted by both ``HDF5File`` (the root of an HDF5 file)
/// and ``HDF5Group`` (a named sub-container within a file). It provides a
/// uniform API for navigating and populating the HDF5 hierarchy regardless of
/// whether the current node is the file root or a nested group.
///
/// All operations are `async throws` and are serialised through the internal
/// HDF5 dispatch queue — see ``HDF5`` for details on the thread-safety model.
public protocol HDF5FileOrGroup {
    /// Creates a new group with the given name directly inside this container.
    ///
    /// - Parameter name: The group name (a single path component, e.g. `"results"`).
    /// - Returns: The newly created ``HDF5Group``.
    /// - Throws: ``HDF5Error/groupCreateFailed(_:)`` if the C library returns an
    ///   error (e.g. a group with that name already exists).
    func createGroup(_ name: String) async throws -> HDF5Group

    /// Opens an existing group with the given name inside this container.
    ///
    /// - Parameter name: The group name (a single path component, e.g. `"results"`).
    /// - Returns: The opened ``HDF5Group``.
    /// - Throws: ``HDF5Error/groupOpenFailed(_:)`` if the group does not exist or
    ///   cannot be opened.
    func openGroup(_ name: String) async throws -> HDF5Group

    /// Creates a new dataset with the given name, datatype, and dataspace inside
    /// this container.
    ///
    /// Create the dataspace first with ``HDF5/createDataspace(dimensions:)``, then
    /// choose a datatype from ``HDF5Datatype`` that matches the Swift element type
    /// you intend to write.
    ///
    /// ```swift
    /// let space   = try await HDF5.createDataspace(dimensions: [1024])
    /// let dataset = try await group.createDataset(
    ///     "signal",
    ///     datatype: HDF5Datatype.float,
    ///     dataspace: space
    /// )
    /// try await dataset.writeDataset(data: floatArray)
    /// ```
    ///
    /// - Parameters:
    ///   - name: The dataset name (a single path component).
    ///   - datatype: The HDF5 type identifier for each element. Use the constants
    ///     from ``HDF5Datatype`` to match your Swift element type.
    ///   - dataspace: The shape descriptor created with
    ///     ``HDF5/createDataspace(dimensions:)``.
    /// - Returns: The newly created ``HDF5Dataset``.
    /// - Throws: ``HDF5Error/datasetCreateFailed(_:)`` if the C library returns an
    ///   error (e.g. a dataset with that name already exists).
    func createDataset(
        _ name: String,
        datatype: hid_t,
        dataspace: HDF5Dataspace
    ) async throws -> HDF5Dataset

    /// Opens an existing dataset with the given name inside this container.
    ///
    /// - Parameter name: The dataset name (a single path component).
    /// - Returns: The opened ``HDF5Dataset``.
    /// - Throws: ``HDF5Error/datasetOpenFailed(_:)`` if the dataset does not exist
    ///   or cannot be opened.
    func openDataset(_ name: String) async throws -> HDF5Dataset
}

protocol HDF5FileOrGroupImpl: HDF5AttributableImpl, HDF5FileOrGroup {
    var id: hid_t { get }
}

extension HDF5FileOrGroupImpl {
    public func createGroup(_ name: String) async throws -> HDF5Group {
        let groupId = try await HDF5.h5Gcreate2(name, self.id)
        return HDF5Group(id: groupId, parent: self)
    }

    public func openGroup(_ name: String) async throws -> HDF5Group {
        let groupId = try await HDF5.h5Gopen2(name, self.id)
        return HDF5Group(id: groupId, parent: self)
    }

    public func createDataset(
        _ name: String,
        datatype: hid_t,
        dataspace: HDF5Dataspace
    ) async throws -> HDF5Dataset {
        let datasetId = try await HDF5.h5Dcreate2(
            parent: self.id,
            name: name,
            datatype: datatype,
            dataspace: dataspace.id
        )
        return HDF5Dataset(id: datasetId, parent: self)
    }

    public func openDataset(_ name: String) async throws -> HDF5Dataset {
        let datasetId = try await HDF5.h5Dopen2(parent: id, name: name)
        return HDF5Dataset(id: datasetId, parent: self)
    }
}

extension HDF5File: HDF5FileOrGroupImpl {}
extension HDF5Group: HDF5FileOrGroupImpl {}
