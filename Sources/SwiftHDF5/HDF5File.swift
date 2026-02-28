

public final class HDF5File: Sendable {
    public let id: hid_t
    
    init(id: hid_t) {
        self.id = id
    }
    
    deinit {
        try? HDF5.h5Fclose(id)
    }
}
