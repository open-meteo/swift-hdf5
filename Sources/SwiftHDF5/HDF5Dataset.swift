public final class HDF5Dataset: Sendable {
    let id: hid_t
    let parent: any HDF5FileOrGroupImpl

    init(id: hid_t, parent: any HDF5FileOrGroupImpl) {
        self.id = id
        self.parent = parent
    }

    public func writeDataset<T: HDF5DatasetType>(data: [T]) async throws {
        try await HDF5.h5Dwrite(dataset: id, data: data)
    }

    public func readDataset<T: HDF5DatasetType>() async throws -> [T] {
        return try await HDF5.readDataset(id)
    }

    public func readDataset<T: HDF5DatasetType>(
        reusing buffer: consuming [T]
    ) async throws -> [T] {
        try await HDF5.readDataset(id, reusing: consume buffer)
    }

    public var space: HDF5Dataspace {
        get async throws {
            let spaceId = try await HDF5.h5Dget_space(dataset: id)
            return HDF5Dataspace(id: spaceId)
        }
    }

    deinit {
        try? HDF5.h5Dclose(id)
    }
}
