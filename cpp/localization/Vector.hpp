#pragma once

#include <cmath>

#define EPSILON 1e-10

inline float Cos(float x)
{
    return (cos(x * M_PI / 180));
}

inline float Sin(float x)
{
    return (sin(x * M_PI / 180));
}

inline float ATan2(float x, float y)
{
    if (fabs(x) < EPSILON && fabs(y) < EPSILON)
        return (0.0);

    return (atan2(x, y) * 180 / M_PI);
}

struct Vector
{
    double x, y;

    Vector(double x = 0.0, double y = 0.0) : x(x), y(y)
    {
    }

    Vector operator-() const
    {
        return Vector(-x, -y);
    }

    Vector operator+(const double d) const
    {
        return Vector(x + d, y + d);
    }

    Vector operator+(const Vector &other) const
    {
        return Vector(x + other.x, y + other.y);
    }

    Vector operator-(const double d) const
    {
        return Vector(x - d, y - d);
    }

    Vector operator-(const Vector &other) const
    {
        return Vector(x - other.x, y - other.y);
    }

    Vector operator*(const double d) const
    {
        return Vector(x * d, y * d);
    }

    Vector operator*(const Vector &other) const
    {
        return Vector(x * other.x, y * other.y);
    }

    Vector operator/(const double d) const
    {
        return Vector(x / d, y / d);
    }

    Vector operator/(const Vector &other) const
    {
        return Vector(x / other.x, y / other.y);
    }

    void operator+=(const double d)
    {
        x += d;
        y += d;
    }

    void operator+=(const Vector &other)
    {
        x += other.x;
        y += other.y;
    }

    void operator-=(const double d)
    {
        x -= d;
        y -= d;
    }

    void operator-=(const Vector &other)
    {
        x -= other.x;
        y -= other.y;
    }

    void operator*=(const double d)
    {
        x *= d;
        y *= d;
    }

    void operator*=(const Vector &other)
    {
        x *= other.x;
        y *= other.y;
    }

    void operator/=(const double d)
    {
        x /= d;
        y /= d;
    }

    void operator/=(const Vector &other)
    {
        x /= other.x;
        y /= other.y;
    }

    bool operator==(const double d)
    {
        return (x == d) && (y == d);
    }

    bool operator==(const Vector &other)
    {
        return (x == other.x) && (y == other.y);
    }

    double distance(const Vector &other)
    {
        return ((*this - other).length());
    }

    double length() const
    {
        return (sqrt(x * x + y * y));
    }

    double crossProduct(const Vector &other)
    {
        return this->x * other.y - this->y * other.x;
    }

    float innerProduct(const Vector &other) const
    {
        return x * other.x + y * other.y;
    }
};
