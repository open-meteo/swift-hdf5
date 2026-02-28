
public final class HDF5Group: Sendable {
    public let id: hid_t
    public let parent: HDF5FileOrGroup
    
    init(id: hid_t, parent: some HDF5FileOrGroup) {
        self.id = id
        self.parent = parent
    }
    
    deinit {
        Task { [id] in
            try? await HDF5.shared.h5Gclose(id)
        }
    }
}
