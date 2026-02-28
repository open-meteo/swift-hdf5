import Foundation

// MARK: - Error Handling

public enum HDF5Error: Error {
    case fileOpenFailed(String)
    case fileCreateFailed(String)
    case fileCloseFailed
    case groupOpenFailed(String)
    case groupCreateFailed(String)
    case groupCloseFailed
    case datasetOpenFailed(String)
    case datasetCreateFailed(String)
    case datasetReadFailed(String)
    case datasetWriteFailed(String)
    case datasetCloseFailed
    case dataspaceCreateFailed
    case dataspaceCloseFailed
    case attributeOpenFailed(String)
    case attributeCreateFailed(String)
    case attributeReadFailed(String)
    case attributeWriteFailed(String)
    case invalidDataType
    case operationFailed(String)
}

// MARK: - File Access Modes


public final class HDF5FileRef: Sendable {
    public let path: String
    public let id: hid_t
    
    init(path: String, id: hid_t) {
        self.path = path
        self.id = id
    }
    
    deinit {
        Task { [id] in
            try? await HDF5.shared.h5Fclose(id)
        }
    }
}

public final class HDF5GroupRef: Sendable {
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

public final class HDF5DatasetRef: Sendable {
    public let id: hid_t
    public let parent: HDF5FileOrGroup
    
    init(id: hid_t, parent: HDF5FileOrGroup) {
        self.id = id
        self.parent = parent
    }
    
    public func writeDataset<T: Sendable>(data: [T]) async throws {
        try await HDF5.shared.h5Dwrite(dataset: id, data: data)
    }
    
    public func readDataset<T: Numeric & Sendable>() async throws -> [T] {
        return try await HDF5.shared.readDataset(id)
    }
    
    public func readDataset<T: Sendable>(
        into buffer: inout [T]
    ) async throws {
        try await HDF5.shared.readDataset(id, into: &buffer)
    }
    
    public func getDatasetSpace() async throws -> HDF5DataspaceRef {
        let spaceId = try await HDF5.shared.h5Dget_space(dataset: id)
        return HDF5DataspaceRef(id: spaceId)
    }
    
    deinit {
        Task { [id] in
            try? await HDF5.shared.h5Dclose(id)
        }
    }
}

public final class HDF5DataspaceRef: Sendable {
    public let id: hid_t
    
    init(id: hid_t) {
        self.id = id
    }
    
    public func getDimensions() async throws -> [UInt64] {
        return try await HDF5.shared.h5Sget_simple_extent_dims(space_id: id)
    }
    
    deinit {
        Task { [id] in
            try? await HDF5.shared.h5Sclose(id)
        }
    }
}

// MARK: - Protocols for parent containers

extension HDF5FileRef: HDF5FileOrGroup {
    public var parentFileOrGroup: (any HDF5FileOrGroup)? {
        return nil
    }
}

extension HDF5GroupRef: HDF5FileOrGroup {
    public var parentFileOrGroup: (any HDF5FileOrGroup)? {
        return self.parent
    }
}


public protocol HDF5FileOrGroup: HDF5Attributable {
    var id: hid_t { get }
    var parentFileOrGroup: HDF5FileOrGroup? { get }
}


extension HDF5FileOrGroup {
    public func createGroup(_ name: String) async throws -> HDF5GroupRef {
        let groupId = try await HDF5.shared.h5Gcreate2(name, self.id)
        return HDF5GroupRef(id: groupId, parent: self)
    }
    
    public func openGroup(_ name: String) async throws -> HDF5GroupRef {
        let groupId = try await HDF5.shared.h5Gopen2(name, self.id)
        return HDF5GroupRef(id: groupId, parent: self)
    }
    
    public func createDataset(
        _ name: String,
        datatype: hid_t,
        dataspace: HDF5DataspaceRef
    ) async throws -> HDF5DatasetRef {
        let datasetId = try await HDF5.shared.h5Dcreate2(parent: self.id, name: name, datatype: datatype, dataspace: dataspace.id)
        guard datasetId >= 0 else { throw HDF5Error.datasetCreateFailed(name) }
        return HDF5DatasetRef(id: datasetId, parent: self)
    }
    
    public func openDataset(_ name: String) async throws -> HDF5DatasetRef {
        let datasetId = try await HDF5.shared.h5Dopen2(parent: id, name: name)
        return HDF5DatasetRef(id: datasetId, parent: self)
    }
}

