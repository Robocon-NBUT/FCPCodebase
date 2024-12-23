#pragma once

#include "Vector3.hpp"

struct Line6
{
    // Polar start and end coordinate
    const Vector3 startp, endp;

    // Cartesian start and end coordinate
    const Vector3 startc, endc;

    const double length;

    Line6(const Vector3 &polar_s, const Vector3 &polar_e)
        : startp(polar_s), endp(polar_e), startc(polar_s.toCartesian()), endc(polar_e.toCartesian()),
          length(startc.dist(endc)){};

    Line6(const Vector3 &cart_s, const Vector3 &cart_e, double length)
        : startp(cart_s.toPolar()), endp(cart_e.toPolar()), startc(cart_s), endc(cart_e), length(length){};

    Line6(const Line6 &obj)
        : startp(obj.startp), endp(obj.endp), startc(obj.startc), endc(obj.endc), length(obj.length){};

    Vector3 linePointClosestToCartPoint(const Vector3 &cp) const
    {

        /**
         * Equation of this line: (we want to find t, such that (cp-p) is perpendicular to this line)
         * p = startc + t*(endc - startc)
         *
         * Let vecp=(cp-start) and vecline=(end-start)
         * Scalar projection of vecp in the direction of vecline:
         *      sp = vecp.vecline/|vecline|
         * Find the ratio t by dividing the scalar projection by the length of vecline:
         *      t = sp / |vecline|
         * So the final expression becomes:
         *      t = vecp.vecline/|vecline|^2
         */

        Vector3 lvec(endc - startc); // this line's vector
        double t = (cp - startc).innerProduct(lvec) / (length * length);

        return startc + lvec * t;
    }

    Vector3 linePointClosestToPolarPoint(const Vector3 &pp) const
    {
        return linePointClosestToCartPoint(pp.toCartesian());
    }

    double lineDistToCartPoint(const Vector3 &cp) const
    {
        return ((cp - startc).crossProduct(cp - endc)).length() / (endc - startc).length();
    }

    double lineDistToPolarPoint(const Vector3 &pp) const
    {
        return lineDistToCartPoint(pp.toCartesian());
    }

    // source: http://geomalgorithms.com/a07-_distance.html#dist3D_Line_to_Line()
    double lineDistToLine(const Line6 &l) const
    {
        Vector3 u = endc - startc;
        Vector3 v = l.endc - l.startc;
        Vector3 w = startc - l.startc;
        double a = u.innerProduct(u); // always >= 0
        double b = u.innerProduct(v);
        double c = v.innerProduct(v); // always >= 0
        double d = u.innerProduct(w);
        double e = v.innerProduct(w);
        double D = a * c - b * b; // always >= 0
        double sc, tc;

        // compute the line parameters of the two closest points
        if (D < 1e-8f)
        { // the lines are almost parallel
            sc = 0.0;
            tc = (b > c ? d / b : e / c); // use the largest denominator
        }
        else
        {
            sc = (b * e - c * d) / D;
            tc = (a * e - b * d) / D;
        }

        // get the difference of the two closest points
        Vector3 dP = w + (u * sc) - (v * tc); // =  L1(sc) - L2(tc)

        return dP.length(); // return the closest distance
    }

    Vector3 segmentPointClosestToCartPoint(const Vector3 &cp) const
    {

        /**
         * Equation of this line: (we want to find t, such that (cp-p) is perpendicular to this line)
         * p = startc + t*(endc - startc)
         *
         * Let vecp=(cp-start) and vecline=(end-start)
         * Scalar projection of vecp in the direction of vecline:
         *      sp = vecp.vecline/|vecline|
         * Find the ratio t by dividing the scalar projection by the length of vecline:
         *      t = sp / |vecline|
         * So the final expression becomes:
         *      t = vecp.vecline/|vecline|^2
         * Since this version requires that p belongs to the line segment, there is an additional restriction:
         *      0 < t < 1
         */

        Vector3 lvec(endc - startc); // this line's vector
        double t = (cp - startc).innerProduct(lvec) / (length * length);

        if (t < 0)
            t = 0;
        else if (t > 1)
            t = 1;

        return startc + lvec * t;
    }

    Vector3 segmentPointClosestToPolarPoint(const Vector3 &pp) const
    {
        return segmentPointClosestToCartPoint(pp.toCartesian());
    }

    double segmentDistToCartPoint(const Vector3 &cp) const
    {

        Vector3 v = endc - startc; // line segment vector
        Vector3 w1 = cp - startc;

        if (w1.innerProduct(v) <= 0)
            return w1.length();

        Vector3 w2 = cp - endc;

        if (w2.innerProduct(v) >= 0)
            return w2.length();

        return v.crossProduct(w1).length() / this->length;
    }

    double segmentDistToPolarPoint(const Vector3 &pp) const
    {
        return segmentDistToCartPoint(pp.toCartesian());
    }

    // source: http://geomalgorithms.com/a07-_distance.html#dist3D_Segment_to_Segment()
    double segmentDistToSegment(const Line6 &other) const
    {

        Vector3 u = endc - startc;
        Vector3 v = other.endc - other.startc;
        Vector3 w = startc - other.startc;
        double a = u.innerProduct(u); // always >= 0
        double b = u.innerProduct(v);
        double c = v.innerProduct(v); // always >= 0
        double d = u.innerProduct(w);
        double e = v.innerProduct(w);
        double D = a * c - b * b; // always >= 0
        double sc, sN, sD = D;    // sc = sN / sD, default sD = D >= 0
        double tc, tN, tD = D;    // tc = tN / tD, default tD = D >= 0

        // compute the line parameters of the two closest points
        if (D < 1e-8f)
        {             // the lines are almost parallel
            sN = 0.0; // force using point start on segment S1
            sD = 1.0; // to prevent possible division by 0.0 later
            tN = e;
            tD = c;
        }
        else
        { // get the closest points on the infinite lines
            sN = (b * e - c * d);
            tN = (a * e - b * d);
            if (sN < 0.0)
            { // sc < 0 => the s=0 edge is visible
                sN = 0.0;
                tN = e;
                tD = c;
            }
            else if (sN > sD)
            { // sc > 1  => the s=1 edge is visible
                sN = sD;
                tN = e + b;
                tD = c;
            }
        }

        if (tN < 0.0)
        { // tc < 0 => the t=0 edge is visible
            tN = 0.0;
            // recompute sc for this edge
            if (-d < 0.0)
                sN = 0.0;
            else if (-d > a)
                sN = sD;
            else
            {
                sN = -d;
                sD = a;
            }
        }
        else if (tN > tD)
        { // tc > 1  => the t=1 edge is visible
            tN = tD;
            // recompute sc for this edge
            if ((-d + b) < 0.0)
                sN = 0;
            else if ((-d + b) > a)
                sN = sD;
            else
            {
                sN = (-d + b);
                sD = a;
            }
        }
        // finally do the division to get sc and tc
        sc = (abs(sN) < 1e-8f ? 0.0 : sN / sD);
        tc = (abs(tN) < 1e-8f ? 0.0 : tN / tD);

        // get the difference of the two closest points
        Vector3 dP = w + (u * sc) - (v * tc); // =  S1(sc) - S2(tc)

        return dP.length(); // return the closest distance
    }

    Vector3 midPointCart() const
    {
        return (startc + endc) / 2;
    }

    Vector3 midPointPolar() const
    {
        return midPointCart().toPolar();
    }

    bool operator==(const Line6 &other) const
    {
        return (startp == other.startp) && (endp == other.endp);
    }

    const Vector3 &get_polar_pt(const int index) const
    {
        if (index == 0)
            return startp;
        return endp;
    }

    const Vector3 &get_cart_pt(const int index) const
    {
        if (index == 0)
            return startc;
        return endc;
    }

    Vector3 get_polar_vector() const
    {
        return endp - startp;
    }

    Vector3 get_cart_vector() const
    {
        return endc - startc;
    }
};
