/// A handle to an open HDF5 group within a file.
///
/// Groups are the directories of the HDF5 hierarchy. Every HDF5 file has an
/// implicit root group (`/`), and named groups can be nested arbitrarily
/// beneath it to organise datasets and attributes.
///
/// Obtain an instance via ``HDF5FileOrGroup/createGroup(_:)`` or
/// ``HDF5FileOrGroup/openGroup(_:)``. The group is closed automatically when
/// the last reference to this object is released.
///
/// `HDF5Group` conforms to ``HDF5FileOrGroup``, so you can nest further groups
/// and datasets inside it, and to ``HDF5Attributable``, so you can attach
/// metadata attributes directly to the group itself.
///
/// ```swift
/// let sensors = try await file.createGroup("sensors")
/// let temp    = try await sensors.createGroup("temperature")
/// try await temp.writeAttribute("unit", value: "Kelvin")
/// ```
public final class HDF5Group: Sendable {
    let id: hid_t
    /// The direct parent container (file or group) that this group lives in.
    let parent: any HDF5FileOrGroupImpl

    init(id: hid_t, parent: any HDF5FileOrGroupImpl) {
        self.id = id
        self.parent = parent
    }

    deinit {
        try? HDF5.h5Gclose(id)
    }
}
