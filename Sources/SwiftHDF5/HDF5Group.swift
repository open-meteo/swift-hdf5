public final class HDF5Group: Sendable {
    let id: hid_t
    let parent: any HDF5FileOrGroupImpl

    init(id: hid_t, parent: any HDF5FileOrGroupImpl) {
        self.id = id
        self.parent = parent
    }

    deinit {
        try? HDF5.h5Gclose(id)
    }
}
