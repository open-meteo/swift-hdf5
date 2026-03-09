import Foundation

// MARK: - Error Handling

public enum HDF5Error: Error, LocalizedError, CustomStringConvertible {
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

    public var description: String {
        switch self {
        case .fileOpenFailed(let path):
            return "Failed to open file at path: '\(path)'"
        case .fileCreateFailed(let path):
            return "Failed to create file at path: '\(path)'"
        case .fileCloseFailed:
            return "Failed to close file."
        case .groupOpenFailed(let name):
            return "Failed to open group: '\(name)'"
        case .groupCreateFailed(let name):
            return "Failed to create group: '\(name)'"
        case .groupCloseFailed:
            return "Failed to close group."
        case .datasetOpenFailed(let name):
            return "Failed to open dataset: '\(name)'"
        case .datasetCreateFailed(let name):
            return "Failed to create dataset: '\(name)'"
        case .datasetReadFailed(let name):
            return "Failed to read from dataset: '\(name)'"
        case .datasetWriteFailed(let name):
            return "Failed to write to dataset: '\(name)'"
        case .datasetCloseFailed:
            return "Failed to close dataset."
        case .dataspaceCreateFailed:
            return "Failed to create dataspace."
        case .dataspaceCloseFailed:
            return "Failed to close dataspace."
        case .attributeOpenFailed(let name):
            return "Failed to open attribute: '\(name)'"
        case .attributeCreateFailed(let name):
            return "Failed to create attribute: '\(name)'"
        case .attributeReadFailed(let name):
            return "Failed to read attribute: '\(name)'"
        case .attributeWriteFailed(let name):
            return "Failed to write attribute: '\(name)'"
        case .invalidDataType:
            return "Encountered an invalid or unsupported HDF5 data type."
        case .operationFailed(let reason):
            return "HDF5 operation failed: \(reason)"
        }
    }

    public var errorDescription: String? {
        return description
    }
}
