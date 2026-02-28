// MARK: - Error Handling

/**
 TODO: errors need to be cleaned up!
 */
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

