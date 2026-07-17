#ifndef INTERACTION_GEOMETRY_HDR
#define INTERACTION_GEOMETRY_HDR

// Geometry helpers, shared by the featurizers.
//
// Positions are stored as float32 but every distance / dihedral is accumulated in double (copy of MDAnalysis convention).

#include <array>
#include <cmath>

using Vec3 = std::array<float, 3>;

inline double dist(const Vec3 &a, const Vec3 &b)
{
    const double dx = static_cast<double>(a[0]) - static_cast<double>(b[0]); // widen to double before subtracting
    const double dy = static_cast<double>(a[1]) - static_cast<double>(b[1]);
    const double dz = static_cast<double>(a[2]) - static_cast<double>(b[2]);
    return std::sqrt(dx * dx + dy * dy + dz * dz); // plain euclidean distance
}

// Dihedral angle p0-p1-p2-p3 in degrees, range (-180, 180].
inline double dihedral_deg(const Vec3 &p0, const Vec3 &p1, const Vec3 &p2,
                           const Vec3 &p3)
{
    const std::array<double, 3> b0{
        static_cast<double>(p0[0]) - static_cast<double>(p1[0]),
        static_cast<double>(p0[1]) - static_cast<double>(p1[1]),
        static_cast<double>(p0[2]) - static_cast<double>(p1[2])};
    std::array<double, 3> b1{
        static_cast<double>(p2[0]) - static_cast<double>(p1[0]),
        static_cast<double>(p2[1]) - static_cast<double>(p1[1]),
        static_cast<double>(p2[2]) - static_cast<double>(p1[2])};
    const std::array<double, 3> b2{
        static_cast<double>(p3[0]) - static_cast<double>(p2[0]),
        static_cast<double>(p3[1]) - static_cast<double>(p2[1]),
        static_cast<double>(p3[2]) - static_cast<double>(p2[2])};

    const double n1 = std::sqrt(b1[0] * b1[0] + b1[1] * b1[1] + b1[2] * b1[2]); // length of the middle bond vector
    if (n1 == 0.0)
        return 0.0; // degenerate geometry, avoid dividing by zero below
    b1[0] /= n1;    // normalize b1 so it can be used as a projection axis
    b1[1] /= n1;
    b1[2] /= n1;

    const double d0 = b0[0] * b1[0] + b0[1] * b1[1] + b0[2] * b1[2]; // component of b0 along b1
    const double d2 = b2[0] * b1[0] + b2[1] * b1[1] + b2[2] * b1[2]; // component of b2 along b1
    const std::array<double, 3> v{b0[0] - d0 * b1[0], b0[1] - d0 * b1[1],
                                  b0[2] - d0 * b1[2]}; // b0 projected onto the plane perpendicular to b1
    const std::array<double, 3> w{b2[0] - d2 * b1[0], b2[1] - d2 * b1[1],
                                  b2[2] - d2 * b1[2]}; // same projection for b2

    const double x = v[0] * w[0] + v[1] * w[1] + v[2] * w[2]; // cosine-like term
    // cross(b1, v) * w
    const double y = (b1[1] * v[2] - b1[2] * v[1]) * w[0] +
                     (b1[2] * v[0] - b1[0] * v[2]) * w[1] +
                     (b1[0] * v[1] - b1[1] * v[0]) * w[2]; // sine-like term, gives the angle its sign
    return std::atan2(y, x) * (180.0 / M_PI);              // convert radians to degrees
}

#endif // INTERACTION_GEOMETRY_HDR
