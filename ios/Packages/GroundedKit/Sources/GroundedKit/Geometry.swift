import Foundation
import simd

/// Camera geometry for anchoring backend regions in the world.
///
/// ARKit images arrive in the sensor's landscape orientation. The app rotates them upright (portrait)
/// before analysis, so returned boxes are portrait-normalized and must be mapped back before casting rays.
public enum Geometry {
    /// Center of a portrait-normalized `[x, y, w, h]` box.
    public static func center(of bbox: [Double]) -> SIMD2<Double> {
        SIMD2(bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
    }

    /// Portrait-normalized point -> sensor-normalized point (image rotated 90° clockwise for display).
    public static func sensorPoint(fromPortrait p: SIMD2<Double>) -> SIMD2<Double> {
        SIMD2(p.y, 1 - p.x)
    }

    /// Sensor-normalized point -> portrait-normalized point; inverse of `sensorPoint(fromPortrait:)`.
    public static func portraitPoint(fromSensor p: SIMD2<Double>) -> SIMD2<Double> {
        SIMD2(1 - p.y, p.x)
    }

    /// World-space ray through a sensor-normalized point, from a camera's pose and intrinsics.
    /// ARKit cameras look down -Z with +Y up, while image rows grow downward.
    public static func ray(
        through point: SIMD2<Double>, resolution: SIMD2<Double>, intrinsics: simd_float3x3, transform: simd_float4x4
    ) -> (origin: SIMD3<Float>, direction: SIMD3<Float>) {
        let pixel = SIMD2<Float>(Float(point.x * resolution.x), Float(point.y * resolution.y))
        let fx = intrinsics[0][0], fy = intrinsics[1][1], cx = intrinsics[2][0], cy = intrinsics[2][1]
        let camera = SIMD4<Float>((pixel.x - cx) / fx, -(pixel.y - cy) / fy, -1, 0)
        let world = transform * camera
        let origin = SIMD3<Float>(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
        return (origin, simd_normalize(SIMD3(world.x, world.y, world.z)))
    }
}
