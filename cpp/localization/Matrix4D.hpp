#pragma once

#include "Vector3.hpp"
#include <vector>

constexpr int M_ROWS = 4;
constexpr int M_COLS = 4;
constexpr int M_LENGTH = M_ROWS * M_COLS;

struct Matrix4D
{
    vector<double> content; // content of the matrix, vector-like

    Matrix4D()
    {
        content = vector<double>(M_LENGTH, 0);
        // 将主对角线元素设为1
        content[0] = content[5] = content[10] = content[15] = 1;
    }

    Matrix4D(const double a[M_LENGTH])
    {
        content = vector<double>(a, a + M_LENGTH);
    }

    Matrix4D(const Matrix4D &other)
    {
        content = other.content;
    }

    Matrix4D(const Vector3 &v)
    {
        content = vector<double>(M_LENGTH, 0);
        content[0] = content[5] = content[10] = content[15] = 1;
        content[3] = v.x;
        content[7] = v.y;
        content[11] = v.z;
    }

    ~Matrix4D()
    {
    }

    void set(unsigned i, double value)
    {
        content[i] = value;
    }

    void set(unsigned i, unsigned j, double value)
    {
        content[M_COLS * i + j] = value;
    }

    double get(unsigned i) const
    {
        return content[i];
    }

    double get(unsigned i, unsigned j) const
    {
        return content[M_COLS * i + j];
    }

    Matrix4D operator+(const Matrix4D &other) const
    {
        vector<double> tmp(M_LENGTH, 0);

        for (int i = 0; i < M_LENGTH; i++)
            tmp[i] = content[i] + other.content[i];

        return Matrix4D(tmp.data());
    }

    Matrix4D operator-(const Matrix4D &other) const
    {
        vector<double> tmp(M_LENGTH, 0);

        for (int i = 0; i < M_LENGTH; i++)
            tmp[i] = content[i] - other.content[i];

        return Matrix4D(tmp.data());
    }

    Matrix4D operator*(const Matrix4D &other) const
    {
        vector<double> tmp(M_LENGTH, 0);

        for (int i = 0; i < M_ROWS; i++)
        {
            for (int j = 0; j < M_COLS; j++)
            {
                tmp[M_COLS * i + j] = 0;
                for (int k = 0; k < M_COLS; k++)
                    tmp[M_COLS * i + j] += content[M_COLS * i + k] * other.content[M_COLS * k + j];
            }
        }

        return Matrix4D(tmp.data());
    }

    Vector3 operator*(const Vector3 &other) const
    {
        double x = content[0] * other.x + content[1] * other.y + content[2] * other.z + content[3];
        double y = content[4] * other.x + content[5] * other.y + content[6] * other.z + content[7];
        double z = content[8] * other.x + content[9] * other.y + content[10] * other.z + content[11];

        return Vector3(x, y, z);
    }

    void operator=(const Matrix4D &other)
    {
        content = other.content;
    }

    bool operator==(const Matrix4D &other) const
    {
        for (int i = 0; i < M_LENGTH; i++)
            if (content[i] != other.content[i])
                return false;
        return true;
    }

    void operator+=(const Matrix4D &other)
    {
        for (int i = 0; i < M_LENGTH; i++)
            content[i] += other.content[i];
    }

    void operator-=(const Matrix4D &other)
    {
        for (int i = 0; i < M_LENGTH; i++)
            content[i] -= other.content[i];
    }

    double &operator[](const unsigned pos)
    {
        return content[pos];
    }

    Vector3 toVector3() const
    {
        double x = get(0, M_COLS - 1);
        double y = get(1, M_COLS - 1);
        double z = get(2, M_COLS - 1);

        return Vector3(x, y, z);
    }

    Matrix4D transpose() const
    {
        Matrix4D result;

        for (int i = 0; i < M_ROWS; i++)
            for (int j = 0; j < M_COLS; j++)
                result.set(j, i, get(i, j));

        return result;
    }

    Matrix4D inverse_tranformation_matrix() const
    {
        Matrix4D inv; //Initialized as identity matrix
        inverse_tranformation_matrix(inv);
        return inv;
    }

    void inverse_tranformation_matrix(Matrix4D &inv) const
    {
        //Rotation
        inv[0] = content[0];    inv[1] = content[4];    inv[2] = content[8];
        inv[4] = content[1];    inv[5] = content[5];    inv[6] = content[9];
        inv[8] = content[2];    inv[9] = content[6];    inv[10] = content[10];

        //Translation
        inv[3] = -content[0]*content[3] - content[4]*content[7] - content[8]*content[11];
        inv[7] = -content[1]*content[3] - content[5]*content[7] - content[9]*content[11];
        inv[11] = -content[2]*content[3] - content[6]*content[7] - content[10]*content[11];
    }

    bool inverse(Matrix4D& inverse_out) const
    {
        const double *m = content.data();
        double inv[16], det;
        int i;

        inv[0] =   m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
        inv[4] =  -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
        inv[8] =   m[4] * m[9]  * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
        inv[12] = -m[4] * m[9]  * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
        inv[1] =  -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
        inv[5] =   m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
        inv[9] =  -m[0] * m[9]  * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
        inv[13] =  m[0] * m[9]  * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
        inv[2] =   m[1] * m[6]  * m[15] - m[1] * m[7]  * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7]  - m[13] * m[3] * m[6];
        inv[6] =  -m[0] * m[6]  * m[15] + m[0] * m[7]  * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7]  + m[12] * m[3] * m[6];
        inv[10] =  m[0] * m[5]  * m[15] - m[0] * m[7]  * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7]  - m[12] * m[3] * m[5];
        inv[14] = -m[0] * m[5]  * m[14] + m[0] * m[6]  * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6]  + m[12] * m[2] * m[5];
        inv[3] =  -m[1] * m[6]  * m[11] + m[1] * m[7]  * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9]  * m[2] * m[7]  + m[9]  * m[3] * m[6];
        inv[7] =   m[0] * m[6]  * m[11] - m[0] * m[7]  * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8]  * m[2] * m[7]  - m[8]  * m[3] * m[6];
        inv[11] = -m[0] * m[5]  * m[11] + m[0] * m[7]  * m[9]  + m[4] * m[1] * m[11] - m[4] * m[3] * m[9]  - m[8]  * m[1] * m[7]  + m[8]  * m[3] * m[5];
        inv[15] =  m[0] * m[5]  * m[10] - m[0] * m[6]  * m[9]  - m[4] * m[1] * m[10] + m[4] * m[2] * m[9]  + m[8]  * m[1] * m[6]  - m[8]  * m[2] * m[5];

        det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

        if (det == 0)
            return false;

        det = 1.0 / det;

        for (i = 0; i < 16; i++)
            inverse_out.set(i, inv[i] * det);

        return true;
    }

    static Matrix4D rotationX(double angle)
    {
        double tmp[M_LENGTH] = {
            1, 0, 0, 0,
            0, Cos(angle), -Sin(angle), 0,
            0, Sin(angle), Cos(angle), 0,
            0, 0, 0, 1};

        return Matrix4D(tmp);
    }

    static Matrix4D rotationY(double angle)
    {
        double tmp[M_LENGTH] = {
            Cos(angle), 0, Sin(angle), 0,
            0, 1, 0, 0,
            -Sin(angle), 0, Cos(angle), 0,
            0, 0, 0, 1};

        return Matrix4D(tmp);
    }

    static Matrix4D rotationZ(double angle)
    {
        double tmp[M_LENGTH] = {
            Cos(angle), -Sin(angle), 0, 0,
            Sin(angle), Cos(angle), 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1};

        return Matrix4D(tmp);
    }

    static Matrix4D rotation(Vector3 axis, double angle)
    {
        double c = Cos(angle);
        double s = Sin(angle);
        double t = 1 - c;

        double x = axis.x;
        double y = axis.y;
        double z = axis.z;

        double tmp[M_LENGTH] = {
            t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0,
            t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0,
            t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0,
            0, 0, 0, 1};

        return Matrix4D(tmp);
    }

    static Matrix4D translation(const Vector3 &v)
    {
        double tmp[M_LENGTH] = {
            1, 0, 0, v.x,
            0, 1, 0, v.y,
            0, 0, 1, v.z,
            0, 0, 0, 1};

        return Matrix4D(tmp);
    }
};
