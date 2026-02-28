// MARK: - Protocols for parent containers

public protocol HDF5FileOrGroup: HDF5Attributable {
    var id: hid_t { get }
}

extension HDF5File: HDF5FileOrGroup {}

extension HDF5Group: HDF5FileOrGroup {}

extension HDF5FileOrGroup {
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
        let datasetId = try await HDF5.h5Dcreate2(parent: self.id, name: name, datatype: datatype, dataspace: dataspace.id)
        guard datasetId >= 0 else { throw HDF5Error.datasetCreateFailed(name) }
        return HDF5Dataset(id: datasetId, parent: self)
    }
    
    public func openDataset(_ name: String) async throws -> HDF5Dataset {
        let datasetId = try await HDF5.h5Dopen2(parent: id, name: name)
        return HDF5Dataset(id: datasetId, parent: self)
    }
}
