public protocol HDF5FileOrGroup {
    func createGroup(_ name: String) async throws -> HDF5Group
    func openGroup(_ name: String) async throws -> HDF5Group
    func createDataset(
        _ name: String,
        datatype: hid_t,
        dataspace: HDF5Dataspace
    ) async throws -> HDF5Dataset
    func openDataset(_ name: String) async throws -> HDF5Dataset
}

protocol HDF5FileOrGroupImpl: HDF5AttributableImpl, HDF5FileOrGroup {
    var id: hid_t { get }
}

extension HDF5FileOrGroupImpl {
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
        let datasetId = try await HDF5.h5Dcreate2(
            parent: self.id,
            name: name,
            datatype: datatype,
            dataspace: dataspace.id
        )
        return HDF5Dataset(id: datasetId, parent: self)
    }

    public func openDataset(_ name: String) async throws -> HDF5Dataset {
        let datasetId = try await HDF5.h5Dopen2(parent: id, name: name)
        return HDF5Dataset(id: datasetId, parent: self)
    }
}

extension HDF5File: HDF5FileOrGroupImpl {}
extension HDF5Group: HDF5FileOrGroupImpl {}
