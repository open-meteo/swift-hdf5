

public final class HDF5File: Sendable {
    public let id: hid_t
    
    init(id: hid_t) {
        self.id = id
    }
    
    deinit {
        Task { [id] in
            try? await HDF5.shared.h5Fclose(id)
        }
    }
}
