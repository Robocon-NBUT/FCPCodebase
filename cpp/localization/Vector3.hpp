#pragma once

#include "Geometry.h"
struct Vector3
{
    double x, y, z;

    Vector3() : x(0), y(0), z(0)
    {
    }
    Vector3(double x, double y, double z) : x(x), y(y), z(z)
    {
    }
    Vector3(const Vector3 &other) : x(other.x), y(other.y), z(other.z)
    {
    }
    Vector3(const Vector &other) : x(other.x), y(other.y), z(0.0)
    {
    }
    ~Vector3()
    {
    }

    double operator[](const int index) const
    {
        double val = 0.0;
        switch (index)
        {
        case 0:
            val = x;
            break;
        case 1:
            val = y;
            break;
        case 2:
            val = z;
            break;
        default:
            break;
        }
        return val;
    }

    Vector3 operator+(const Vector3 &other) const
    {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }
    Vector3 operator-(const Vector3 &other) const
    {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }
    Vector3 operator-() const
    {
        return Vector3() - *this;
    }
    Vector3 operator*(const Vector3 &other) const
    {
        return Vector3(x * other.x, y * other.y, z * other.z);
    }
    Vector3 operator/(const Vector3 &other) const
    {
        return Vector3(x / other.x, y / other.y, z / other.z);
    }
    bool operator==(const Vector3 &other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }
    Vector3 operator/(double factor) const
    {
        return Vector3(x / factor, y / factor, z / factor);
    }
    Vector3 operator+(double factor) const
    {
        return Vector3(x + factor, y + factor, z + factor);
    }
    Vector3 operator%(double factor) const
    {
        return Vector3(fmod(x, factor), fmod(y, factor), fmod(z, factor));
    }
    Vector3 operator*(double factor) const
    {
        return Vector3(x * factor, y * factor, z * factor);
    }
    Vector3 operator+=(const Vector3 &other)
    {
        x += other.x;
        y += other.y;
        z += other.z;

        return *this;
    }
    Vector3 operator+=(double factor)
    {
        x += factor;
        y += factor;
        z += factor;

        return *this;
    }
    Vector3 operator-=(const Vector3 &other)
    {
        x -= other.x;
        y -= other.y;
        z -= other.z;

        return *this;
    }
    Vector3 operator-=(double factor)
    {
        x -= factor;
        y -= factor;
        z -= factor;

        return *this;
    }
    Vector3 operator*=(const Vector3 &other)
    {
        x *= other.x;
        y *= other.y;
        z *= other.z;

        return *this;
    }
    Vector3 operator*=(double factor)
    {
        x *= factor;
        y *= factor;
        z *= factor;

        return *this;
    }
    Vector3 operator/=(const Vector3 &other)
    {
        x /= other.x;
        y /= other.y;
        z /= other.z;

        return *this;
    }
    Vector3 operator/=(double factor)
    {
        x /= factor;
        y /= factor;
        z /= factor;

        return *this;
    }

    double innerProduct(const Vector3 &other) const
    {
        return x * other.x + y * other.y + z * other.z;
    }

    Vector3 crossProduct(const Vector3 &other) const
    {
        return Vector3(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
    }

    double length() const
    {
        return sqrt(x * x + y * y + z * z);
    }

    Vector3 normalize(double len = 1) const
    {
        return (*this) * (len / this->length());
    }

    Vector3 toCartesian() const
    {
        return Vector3(x * Cos(z) * Cos(y), x * Cos(z) * Sin(y), x * Sin(z));
    }

    Vector3 toPolar() const
    {
        double r = length();
        double theta = ATan2(y, x);
        double phi = ATan2(z, sqrt(x * x + y * y));
        return Vector3(r, theta, phi);
    }

    double dist(const Vector3 &other) const
    {
        return (*this - other).length();
    }

    Vector to2d() const
    {
        return Vector(x, y);
    }

    static Vector3 determineMidpoint(Vector3 a, Vector3 b)
    {
        return (a + b) / 2;
    }
};
