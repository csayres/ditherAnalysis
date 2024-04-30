#include <math.h>
#include <stdio.h>

// compile on max os like gcc -shared -fPIC -o gausslib.so gausslib.c
// help: https://stackoverflow.com/questions/61606342/passing-arguments-to-scipy-lowlevelcallable-while-using-functions-in-c

const double pi = 3.14159265358979323846;

double bivariateGaussian(double r, double theta, double amp, double sigma, double starx, double stary, double fibx, double fiby){
    double x = r * cos(theta);
    double y = r * sin(theta);
    double A = amp * (1.0 / 2 * pi * sigma * sigma);
    double B = ((x+starx-fibx)/sigma) * ((x+starx-fibx)/sigma);
    double C = ((y+stary-fiby)/sigma) * ((y+stary-fiby)/sigma);
    return r * A * exp(-0.5 * (B+C));
}

double f(int n, double *x, void *user_data){
    // cast user_data to double
    double *args = (double *)user_data;
    return bivariateGaussian(
        x[0], x[1], args[0], args[1], args[2], args[3], args[4], args[5]
    );
}