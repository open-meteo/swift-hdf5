
public final class HDF5Dataset: Sendable {
    public let id: hid_t
    public let parent: HDF5FileOrGroup
    
    init(id: hid_t, parent: HDF5FileOrGroup) {
        self.id = id
        self.parent = parent
    }
    
    public func writeDataset<T: Sendable>(data: [T]) async throws {
        try await HDF5.h5Dwrite(dataset: id, data: data)
    }
    
    public func readDataset<T: Numeric & Sendable>() async throws -> [T] {
        return try await HDF5.readDataset(id)
    }
    
//    public func readDataset<T: Sendable>(
//        into buffer: inout [T]
//    ) async throws {
//        try await HDF5.readDataset(id, into: &buffer)
//    }
    
    public func getDatasetSpace() async throws -> HDF5Dataspace {
        let spaceId = try await HDF5.h5Dget_space(dataset: id)
        return HDF5Dataspace(id: spaceId)
    }
    
    deinit {
        try? HDF5.h5Dclose(id)
    }
}
