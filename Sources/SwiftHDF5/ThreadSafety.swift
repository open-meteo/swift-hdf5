extension MainActor {
    public func synchronized<T: Sendable>(_ block: @Sendable () throws -> T) rethrows -> T {
        try block()
    }
}
