#pragma once

extern float ball_pos_pred[600]; // ball position   (x,y) prediction for 300*0.02s = 6s 
extern float ball_vel_pred[600]; // ball velocoty   (x,y) prediction for 300*0.02s = 6s 
extern float ball_spd_pred[300]; // ball linear speed (s) prediction for 300*0.02s = 6s 
extern int pos_pred_len;

extern void get_intersection_with_ball(double x, double y, double max_robot_sp_per_step, double ball_pos[], double ball_pos_len,
                                       double &ret_x, double &ret_y, double &ret_d);
extern void predict_rolling_ball_pos_vel_spd(double bx, double by, double vx, double vy);
