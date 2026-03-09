import CHDF5

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
