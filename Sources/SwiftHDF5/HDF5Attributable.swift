
public protocol HDF5Attributable: Sendable {
    var id: hid_t { get }
}

extension HDF5Attributable {
    public func writeAttribute<T: Sendable>(
        _ name: String,
        value: T,
        datatype: hid_t
    ) async throws {
        try await HDF5.shared.writeAttribute(name, on: id, value: value, datatype: datatype)
    }
    
    public func readAttribute<T: Sendable>(_ name: String) async throws -> T {
        try await HDF5.shared.readAttribute(name, from: id)
    }
}

extension HDF5FileRef: HDF5Attributable, HDF5Parent {
    public var parentId: hid_t { self.id }
    public var parentName: String { "" }
    public var parentFileId: hid_t { self.id }
}

extension HDF5GroupRef: HDF5Attributable, HDF5Parent {
    public var parentId: hid_t { self.id }
    public var parentName: String { self.name }
    public var parentFileId: hid_t { self.fileId }
}

extension HDF5DatasetRef: HDF5Attributable {}
extension HDF5DataspaceRef: HDF5Attributable {}
