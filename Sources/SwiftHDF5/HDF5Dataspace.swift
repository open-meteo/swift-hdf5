
public final class HDF5Dataspace: Sendable {
    public let id: hid_t
    
    init(id: hid_t) {
        self.id = id
    }
    
    public func getDimensions() async throws -> [UInt64] {
        return try await HDF5.h5Sget_simple_extent_dims(space_id: id)
    }
    
    deinit {
        try? HDF5.h5Sclose(id)
    }
}
