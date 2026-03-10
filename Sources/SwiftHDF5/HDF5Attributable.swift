/// A type that can carry HDF5 attributes and expose its HDF5 object name.
///
/// Attributes are small pieces of metadata attached directly to an HDF5 object
/// (file, group, or dataset). They are distinct from datasets: they are not
/// chunked or compressed and are intended for descriptive metadata rather than
/// bulk data storage.
///
/// Conforming types: ``HDF5File``, ``HDF5Group``, ``HDF5Dataset``.
public protocol HDF5Attributable {
    /// The HDF5 object path of this object within its file, e.g. `"/data/measurements"`.
    ///
    /// - Throws: ``HDF5Error/operationFailed(_:)`` if the name cannot be retrieved.
    var name: String { get async throws }

    /// The absolute filesystem path of the HDF5 file that contains this object.
    ///
    /// - Throws: ``HDF5Error/operationFailed(_:)`` if the file name cannot be retrieved.
    var fileName: String { get async throws }

    /// Writes a scalar attribute named `name` on this object.
    ///
    /// The HDF5 datatype is inferred automatically from the Swift type of `value`.
    /// Supported types are those conforming to `HDF5AttributeType`: the numeric
    /// types listed in ``HDF5Datatype`` as well as `String`.
    ///
    /// ```swift
    /// try await dataset.writeAttribute("units", value: "Kelvin")
    /// try await dataset.writeAttribute("scale_factor", value: Double(0.01))
    /// ```
    ///
    /// - Parameters:
    ///   - name: The attribute name. Must be unique on this object.
    ///   - value: The scalar value to store.
    /// - Throws: ``HDF5Error/attributeCreateFailed(_:)`` or
    ///   ``HDF5Error/attributeWriteFailed(_:)`` on failure.
    func writeAttribute<T: HDF5AttributeType>(
        _ name: String,
        value: T
    ) async throws

    /// Reads a scalar attribute named `name` from this object.
    ///
    /// The expected HDF5 datatype is inferred from the return type annotation.
    ///
    /// ```swift
    /// let units: String  = try await dataset.readAttribute("units")
    /// let scale: Double  = try await dataset.readAttribute("scale_factor")
    /// ```
    ///
    /// - Parameter name: The attribute name to look up.
    /// - Returns: The attribute value cast to `T`.
    /// - Throws: ``HDF5Error/attributeOpenFailed(_:)`` if the attribute does not
    ///   exist, ``HDF5Error/attributeReadFailed(_:)`` if the read fails, or
    ///   ``HDF5Error/invalidDataType`` if the stored type cannot be retrieved.
    func readAttribute<T: HDF5AttributeType>(_ name: String) async throws -> T
}

protocol HDF5AttributableImpl: Sendable, HDF5Attributable {
    var id: hid_t { get }
}

extension HDF5AttributableImpl {
    public var name: String {
        get async throws {
            return try await HDF5.h5Iget_name(id: id)
        }
    }

    public var fileName: String {
        get async throws {
            return try await HDF5.h5Fget_name(id: id)
        }
    }

    /// Writes an attribute whose HDF5 datatype is inferred automatically from the
    /// Swift type of `value`.
    ///
    /// ```swift
    /// try await dataset.writeAttribute("scale", value: Double(1.5))
    /// try await dataset.writeAttribute("label", value: "temperature")
    /// ```
    public func writeAttribute<T: HDF5AttributeType>(
        _ name: String,
        value: T
    ) async throws {
        try await HDF5.writeAttribute(name, on: id, value: value)
    }

    /// Reads an attribute, inferring the expected HDF5 datatype from the return type.
    ///
    /// ```swift
    /// let scale: Double = try await dataset.readAttribute("scale")
    /// let label: String = try await dataset.readAttribute("label")
    /// ```
    public func readAttribute<T: HDF5AttributeType>(_ name: String) async throws -> T {
        try await HDF5.readAttribute(name, from: id)
    }

}

extension HDF5Group: HDF5AttributableImpl {}
extension HDF5Dataset: HDF5AttributableImpl {}
