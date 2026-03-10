import CHDF5

/// Native HDF5 type identifiers for use with ``HDF5FileOrGroup/createDataset(_:datatype:dataspace:)``.
///
/// Each property returns the HDF5 identifier (`hid_t`) for the corresponding
/// platform-native C type. The mapping to Swift types is:
///
/// | Property       | Swift type  |
/// |----------------|-------------|
/// | ``int8``       | `Int8`      |
/// | ``int16``      | `Int16`     |
/// | ``int32``      | `Int32`     |
/// | ``int64``      | `Int64`     |
/// | ``uint8``      | `UInt8`     |
/// | ``uint16``     | `UInt16`    |
/// | ``uint32``     | `UInt32`    |
/// | ``uint64``     | `UInt64`    |
/// | ``float``      | `Float`     |
/// | ``double``     | `Double`    |
/// | ``char``       | `CChar`     |
enum HDF5Datatype {
    /// The native signed 8-bit integer type (`H5T_NATIVE_INT8`). Corresponds to Swift's `Int8`.
    static var int8: hid_t { hdf5_get_native_int8() }
    /// The native signed 16-bit integer type (`H5T_NATIVE_INT16`). Corresponds to Swift's `Int16`.
    static var int16: hid_t { hdf5_get_native_int16() }
    /// The native signed 32-bit integer type (`H5T_NATIVE_INT32`). Corresponds to Swift's `Int32`.
    static var int32: hid_t { hdf5_get_native_int32() }
    /// The native signed 64-bit integer type (`H5T_NATIVE_INT64`). Corresponds to Swift's `Int64`.
    static var int64: hid_t { hdf5_get_native_int64() }
    /// The native unsigned 8-bit integer type (`H5T_NATIVE_UINT8`). Corresponds to Swift's `UInt8`.
    static var uint8: hid_t { hdf5_get_native_uint8() }
    /// The native unsigned 16-bit integer type (`H5T_NATIVE_UINT16`). Corresponds to Swift's `UInt16`.
    static var uint16: hid_t { hdf5_get_native_uint16() }
    /// The native unsigned 32-bit integer type (`H5T_NATIVE_UINT32`). Corresponds to Swift's `UInt32`.
    static var uint32: hid_t { hdf5_get_native_uint32() }
    /// The native unsigned 64-bit integer type (`H5T_NATIVE_UINT64`). Corresponds to Swift's `UInt64`.
    static var uint64: hid_t { hdf5_get_native_uint64() }
    /// The native 32-bit floating-point type (`H5T_NATIVE_FLOAT`). Corresponds to Swift's `Float`.
    static var float: hid_t { hdf5_get_native_float() }
    /// The native 64-bit floating-point type (`H5T_NATIVE_DOUBLE`). Corresponds to Swift's `Double`.
    static var double: hid_t { hdf5_get_native_double() }
    /// The native C `char` type (`H5T_NATIVE_CHAR`). Corresponds to Swift's `CChar`.
    static var char: hid_t { hdf5_get_native_char() }
}

/// A type that can be stored in and read back from an HDF5 dataset.
///
/// `String` deliberately does **not** conform: variable-length string datasets
/// require a different read/write path (vlen reclaim) that is not supported by
/// the simple contiguous array API.
///
/// All Swift fixed-width integer types, `Float`, and `Double` conform out of
/// the box.
public protocol HDF5DatasetType: HDF5AttributeType {
    /// A zero-like value used to fill the pre-allocated read buffer.
    static var defaultValue: Self { get }
    /// The HDF5 type class (e.g. `H5T_INTEGER`, `H5T_FLOAT`) for this type.
    /// Used to validate the on-disk type before reading.
    static var hdf5TypeClass: H5T_class_t { get }
    /// The size in bytes of this type's HDF5 representation.
    /// Used together with ``hdf5TypeClass`` to validate the on-disk type.
    static var hdf5TypeSize: Int { get }
}

extension HDF5DatasetType {
    /// Returns a human-readable description of an HDF5 type identifier, suitable
    /// for use in error messages. Describes the type class and byte width.
    static func hdf5TypeDescription(_ typeId: hid_t) -> String {
        let size = H5Tget_size(typeId)
        switch H5Tget_class(typeId) {
        case H5T_INTEGER:
            let sign = H5Tget_sign(typeId)
            let prefix = (sign == H5T_SGN_NONE) ? "UInt" : "Int"
            return "\(prefix)\(size * 8)"
        case H5T_FLOAT:
            return "Float\(size * 8)"
        default:
            return "unknown(class=\(H5Tget_class(typeId).rawValue), size=\(size))"
        }
    }
}

extension Int8: HDF5DatasetType {
    public static var defaultValue: Int8 { 0 }
    public static var hdf5TypeClass: H5T_class_t { H5T_INTEGER }
    public static var hdf5TypeSize: Int { 1 }
}

extension Int16: HDF5DatasetType {
    public static var defaultValue: Int16 { 0 }
    public static var hdf5TypeClass: H5T_class_t { H5T_INTEGER }
    public static var hdf5TypeSize: Int { 2 }
}

extension Int32: HDF5DatasetType {
    public static var defaultValue: Int32 { 0 }
    public static var hdf5TypeClass: H5T_class_t { H5T_INTEGER }
    public static var hdf5TypeSize: Int { 4 }
}

extension Int64: HDF5DatasetType {
    public static var defaultValue: Int64 { 0 }
    public static var hdf5TypeClass: H5T_class_t { H5T_INTEGER }
    public static var hdf5TypeSize: Int { 8 }
}

extension UInt8: HDF5DatasetType {
    public static var defaultValue: UInt8 { 0 }
    public static var hdf5TypeClass: H5T_class_t { H5T_INTEGER }
    public static var hdf5TypeSize: Int { 1 }
}

extension UInt16: HDF5DatasetType {
    public static var defaultValue: UInt16 { 0 }
    public static var hdf5TypeClass: H5T_class_t { H5T_INTEGER }
    public static var hdf5TypeSize: Int { 2 }
}

extension UInt32: HDF5DatasetType {
    public static var defaultValue: UInt32 { 0 }
    public static var hdf5TypeClass: H5T_class_t { H5T_INTEGER }
    public static var hdf5TypeSize: Int { 4 }
}

extension UInt64: HDF5DatasetType {
    public static var defaultValue: UInt64 { 0 }
    public static var hdf5TypeClass: H5T_class_t { H5T_INTEGER }
    public static var hdf5TypeSize: Int { 8 }
}

extension Float: HDF5DatasetType {
    public static var defaultValue: Float { .nan }
    public static var hdf5TypeClass: H5T_class_t { H5T_FLOAT }
    public static var hdf5TypeSize: Int { 4 }
}

extension Double: HDF5DatasetType {
    public static var defaultValue: Double { .nan }
    public static var hdf5TypeClass: H5T_class_t { H5T_FLOAT }
    public static var hdf5TypeSize: Int { 8 }
}

/// A type that can be mapped directly to a native HDF5 datatype.
///
/// Conforming to this protocol allows Swift types to be written and read as HDF5
/// attributes and datasets without the caller needing to supply a `datatype: hid_t`
/// argument manually. The mapping is resolved at compile time, preventing accidental
/// mismatches between a Swift value and its HDF5 representation.
///
/// All Swift fixed-width integer types, `Float`, `Double`, and `String` conform
/// out of the box. Extend this protocol to add support for additional types.
public protocol HDF5AttributeType: Sendable {
    /// The HDF5 native type identifier that corresponds to this Swift type.
    static var hdf5TypeId: hid_t { get }
}

extension Int8: HDF5AttributeType {
    public static var hdf5TypeId: hid_t { hdf5_get_native_int8() }
}

extension Int16: HDF5AttributeType {
    public static var hdf5TypeId: hid_t { hdf5_get_native_int16() }
}

extension Int32: HDF5AttributeType {
    public static var hdf5TypeId: hid_t { hdf5_get_native_int32() }
}

extension Int64: HDF5AttributeType {
    public static var hdf5TypeId: hid_t { hdf5_get_native_int64() }
}

extension UInt8: HDF5AttributeType {
    public static var hdf5TypeId: hid_t { hdf5_get_native_uint8() }
}

extension UInt16: HDF5AttributeType {
    public static var hdf5TypeId: hid_t { hdf5_get_native_uint16() }
}

extension UInt32: HDF5AttributeType {
    public static var hdf5TypeId: hid_t { hdf5_get_native_uint32() }
}

extension UInt64: HDF5AttributeType {
    public static var hdf5TypeId: hid_t { hdf5_get_native_uint64() }
}

extension Float: HDF5AttributeType {
    public static var hdf5TypeId: hid_t { hdf5_get_native_float() }
}

extension Double: HDF5AttributeType {
    public static var hdf5TypeId: hid_t { hdf5_get_native_double() }
}

/// `String` uses a variable-length C string type derived from `H5T_C_S1`.
/// The type identifier is created fresh each call; callers inside the HDF5
/// serial queue are responsible for closing it with `H5Tclose` after use.
extension String: HDF5AttributeType {
    public static var hdf5TypeId: hid_t {
        let typeId = H5Tcopy(hdf5_get_c_s1())
        H5Tset_size(typeId, hdf5_variable_length_string_size())
        H5Tset_strpad(typeId, H5T_STR_NULLTERM)
        H5Tset_cset(typeId, H5T_CSET_UTF8)
        return typeId
    }
}
