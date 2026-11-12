// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "GroundedKit",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [.library(name: "GroundedKit", targets: ["GroundedKit"])],
    targets: [
        .target(name: "GroundedKit"),
        .testTarget(name: "GroundedKitTests", dependencies: ["GroundedKit"]),
    ]
)
