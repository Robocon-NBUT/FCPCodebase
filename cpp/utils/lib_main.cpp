#include "a_star.h"
#include "ball_predictor.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

using namespace pybind11::literals;
namespace py = pybind11;
using namespace std;

py::array_t<double> astar_compute(py::array_t<double> parameters)
{
    auto buff = parameters.request();
    auto ptr = (double *)buff.ptr;
    int len = buff.shape[0];

    astar(ptr, len);

    py::array_t<float> retval(final_path_size);

    auto ret_buff = retval.request();
    auto ret_ptr = (float *)ret_buff.ptr;

    for (int i = 0; i < final_path_size; i++)
    {
        ret_ptr[i] = final_path[i];
    }
    return retval;
}

py::array_t<double> predict_rolling_ball(py::array_t<double> parameters)
{
    auto buff = parameters.request();
    auto ptr = (double *)buff.ptr;
    int len = buff.shape[0];

    predict_rolling_ball_pos_vel_spd(ptr[0], ptr[1], ptr[2], ptr[3]);

    py::array_t<double> retval(pos_pred_len + pos_pred_len + pos_pred_len / 2);
    auto ret_buff = retval.request();
    auto ret_ptr = (double *)ret_buff.ptr;

    for (int i = 0; i < pos_pred_len; i++)
    {
        ret_ptr[i] = ball_pos_pred[i];
    }
    ret_ptr += pos_pred_len;
    for (int i = 0; i < pos_pred_len; i++)
    {
        ret_ptr[i] = ball_vel_pred[i];
    }
    ret_ptr += pos_pred_len;
    for (int i = 0; i < pos_pred_len / 2; i++)
    {
        ret_ptr[i] = ball_spd_pred[i];
    }
    return retval;
}

py::array_t<double> get_intersection(py::array_t<double> parameters)
{
    auto buff = parameters.request();
    auto ptr = (double *)buff.ptr;
    int len = buff.shape[0];

    double ret_x, ret_y, ret_d;

    get_intersection_with_ball(ptr[0], ptr[1], ptr[2], ptr + 3, len - 3, ret_x, ret_y, ret_d);

    py::array_t<double> retval(3);
    auto ret_buff = retval.request();
    auto ret_ptr = (double *)ret_buff.ptr;

    ret_ptr[0] = ret_x;
    ret_ptr[1] = ret_y;
    ret_ptr[2] = ret_d;

    return retval;
}

PYBIND11_MODULE(utils, m)
{
    m.doc() = "C++ Module Extensions";

    m.def("astar_compute", &astar_compute, "Compute the best path with a-star", "parameters"_a);
    m.def("predict_rolling_ball", &predict_rolling_ball, "Predict rolling ball", "parameters"_a);
    m.def("get_intersection", &get_intersection, "Get point of intersection with moving ball", "parameters"_a);
}
