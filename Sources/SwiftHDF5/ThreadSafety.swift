import Synchronization

/// Global synchronization for HDF5 C API calls
/// HDF5 is not thread-safe, so all access must be serialized
enum HDF5Lock {
    private static let mutex = Mutex<Void>(())

    /// Execute a block while holding the exclusive HDF5 lock
    ///
    /// - Parameter block: The code to execute with exclusive access to HDF5
    /// - Returns: The result of the block
    /// - Throws: Any error thrown by the block
    @inline(__always)
    static func synchronized<T>(_ block: () throws -> T) rethrows -> T {
        try mutex.withLock { _ in
            try block()
        }
    }

    /// Execute a block while holding the exclusive HDF5 lock
    ///
    /// Version for non-throwing closures (avoids unnecessary overhead)
    @inline(__always)
    static func synchronized<T>(_ block: () -> T) -> T {
        mutex.withLock { _ in
            block()
        }
    }
}
