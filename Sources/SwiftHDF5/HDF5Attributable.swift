public protocol HDF5Attributable {
    var name: String { get async throws }
    var fileName: String { get async throws }
    func writeAttribute<T: HDF5AttributeType>(
        _ name: String,
        value: T
    ) async throws
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
