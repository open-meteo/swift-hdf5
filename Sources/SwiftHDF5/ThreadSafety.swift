public actor HDF5ActorLock {
    public static let shared = HDF5ActorLock()

    public func synchronized<T: Sendable>(_ block: @Sendable () throws -> T) rethrows -> T {
        try block()
    }
}
