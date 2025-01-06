#pragma once

#include <cmath>
class FieldNoise
{
  public:
    static double log_prob_r(double d, double r)
    {
        double c1 = 100.0 * ((r - 0.005) / d - 1);
        double c2 = 100.0 * ((r + 0.005) / d - 1);
        return log_prob_normal_distribution(0, 0.0965, c1, c2);
    }

    static double log_prob_h(double h, double phi)
    {
        double c1 = phi - 0.005 - h;
        double c2 = phi + 0.005 - h;
        return log_prob_normal_distribution(0, 0.1225, c1, c2);
    }

    static double log_prob_v(double v, double theta)
    {
        double c1 = theta - 0.005 - v;
        double c2 = theta + 0.005 - v;
        return log_prob_normal_distribution(0, 0.1480, c1, c2);
    }

  private:
    FieldNoise(){}; // Disable construction

    static double log_prob_normal_distribution(double mean, double std, double interval1, double interval2)
    {
        const double std2 = std * std::sqrt(2);
        double erf1_x = (mean - interval1) / std2; // lowest interval, highest expression
        double erf2_x = (mean - interval2) / std2; // highest interval, lowest expression

        const double log05 = log(0.5);

        // If they have different sign or |erf1_x|<1 || |erf2_x|<1
        if (fabs(erf1_x) < 2 || fabs(erf2_x) < 2 || ((erf1_x > 0) ^ (erf2_x > 0)))
        {
            // same but faster than log( 0.5 * (erf(erf1_x) - erf(erf2_x)) )
            return log(erf(erf1_x) - erf(erf2_x)) + log05;
        }

        double erf1 = erf_aux(erf1_x);
        double erf2 = erf_aux(erf2_x);

        // These operations are described in the documentation of erf_aux()
        if (erf1_x > 0)
        { // both are positive
            return log(1.0 - exp(erf1 - erf2)) + erf2 + log05;
        }
        else
        { // both are negative
            return log(1.0 - exp(erf2 - erf1)) + erf1 + log05;
        }
    }

    static double erf_aux(double a)
    {
        double r, s, t, u;

        t = fabs(a);
        s = a * a;

        r = fma(-5.6271698458222802e-018, t, 4.8565951833159269e-016);
        u = fma(-1.9912968279795284e-014, t, 5.1614612430130285e-013);
        r = fma(r, s, u);
        r = fma(r, t, -9.4934693735334407e-012);
        r = fma(r, t, 1.3183034417266867e-010);
        r = fma(r, t, -1.4354030030124722e-009);
        r = fma(r, t, 1.2558925114367386e-008);
        r = fma(r, t, -8.9719702096026844e-008);
        r = fma(r, t, 5.2832013824236141e-007);
        r = fma(r, t, -2.5730580226095829e-006);
        r = fma(r, t, 1.0322052949682532e-005);
        r = fma(r, t, -3.3555264836704290e-005);
        r = fma(r, t, 8.4667486930270974e-005);
        r = fma(r, t, -1.4570926486272249e-004);
        r = fma(r, t, 7.1877160107951816e-005);
        r = fma(r, t, 4.9486959714660115e-004);
        r = fma(r, t, -1.6221099717135142e-003);
        r = fma(r, t, 1.6425707149019371e-004);
        r = fma(r, t, 1.9148914196620626e-002);
        r = fma(r, t, -1.0277918343487556e-001);
        r = fma(r, t, -6.3661844223699315e-001);
        r = fma(r, t, -1.2837929411398119e-001);
        r = fma(r, t, -t);

        return r;
    }
};
