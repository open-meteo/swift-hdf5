#ifndef CHDF5_SHIM_H
#define CHDF5_SHIM_H

#include <hdf5.h>

// HDF5 constants wrapper functions (needed because HDF5 macros contain runtime initialization)

// File access flags
static inline unsigned int hdf5_get_f_acc_rdonly(void) { return H5F_ACC_RDONLY; }
static inline unsigned int hdf5_get_f_acc_rdwr(void) { return H5F_ACC_RDWR; }
static inline unsigned int hdf5_get_f_acc_trunc(void) { return H5F_ACC_TRUNC; }
static inline unsigned int hdf5_get_f_acc_excl(void) { return H5F_ACC_EXCL; }

// Dataspace types
static inline H5S_class_t hdf5_get_s_scalar(void) { return H5S_SCALAR; }
static inline H5S_class_t hdf5_get_s_simple(void) { return H5S_SIMPLE; }
static inline H5S_class_t hdf5_get_s_null(void) { return H5S_NULL; }

// Property list identifiers
static inline hid_t hdf5_get_p_default(void) { return H5P_DEFAULT; }

// Dataspace selection
static inline hid_t hdf5_get_s_all(void) { return H5S_ALL; }

// Native type identifiers
// These are needed because H5T_NATIVE_* are macros that Swift cannot import directly
static inline hid_t hdf5_get_native_int8(void) { return H5T_NATIVE_INT8; }
static inline hid_t hdf5_get_native_int16(void) { return H5T_NATIVE_INT16; }
static inline hid_t hdf5_get_native_int32(void) { return H5T_NATIVE_INT32; }
static inline hid_t hdf5_get_native_int64(void) { return H5T_NATIVE_INT64; }
static inline hid_t hdf5_get_native_uint8(void) { return H5T_NATIVE_UINT8; }
static inline hid_t hdf5_get_native_uint16(void) { return H5T_NATIVE_UINT16; }
static inline hid_t hdf5_get_native_uint32(void) { return H5T_NATIVE_UINT32; }
static inline hid_t hdf5_get_native_uint64(void) { return H5T_NATIVE_UINT64; }
static inline hid_t hdf5_get_native_float(void) { return H5T_NATIVE_FLOAT; }
static inline hid_t hdf5_get_native_double(void) { return H5T_NATIVE_DOUBLE; }
static inline hid_t hdf5_get_native_char(void) { return H5T_NATIVE_CHAR; }

// String type helpers
static inline hid_t hdf5_get_c_s1(void) { return H5T_C_S1; }
// H5T_VARIABLE is a macro that cannot be imported directly into Swift
static inline size_t hdf5_variable_length_string_size(void) { return H5T_VARIABLE; }

// Variable-length memory reclaim compatibility shim.
// H5Treclaim was introduced in HDF5 1.12.0 as a direct replacement for the
// deprecated H5Dvlen_reclaim.
static inline herr_t hdf5_vlen_reclaim(hid_t type_id, hid_t space_id, hid_t plist_id, void *buf) {
#if H5_VERSION_GE(1, 12, 0)
    return H5Treclaim(type_id, space_id, plist_id, buf);
#else
    return H5Dvlen_reclaim(type_id, space_id, plist_id, buf);
#endif
}

#endif // CHDF5_SHIM_H
