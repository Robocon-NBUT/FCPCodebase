/**
 * FILENAME:     World
 * DESCRIPTION:  World data from Python
 * AUTHOR:       Miguel Abreu (m.abreu@fe.up.pt)
 * DATE:         2021
 */

#pragma once
#include "Line6.hpp"
#include "Matrix4D.hpp"
#include "Singleton.h"
#include "Vector3.hpp"
#include <array>
#include <vector>

using namespace std;

class World
{
    friend class Singleton<World>;

  private:
    World(){};

  public:
    // Feet variables: (0) left,  (1) right
    bool foot_touch[2];              // is foot touching ground
    Vector3 foot_contact_rel_pos[2]; // foot_transform * translation(foot_contact_pt)

    bool ball_seen;
    Vector3 ball_rel_pos_cart;
    Vector3 ball_cheat_abs_cart_pos;
    Vector3 my_cheat_abs_cart_pos;

    struct sLMark
    {
        bool seen;
        bool isCorner;
        Vector3 pos;
        Vector3 rel_pos;
    };

    sLMark landmark[8];

    struct sLine
    {
        Vector3 start, end;
        sLine(const Vector3 &s, const Vector3 &e) : start(s), end(e){};
    };

    vector<sLine> lines_polar;
};

typedef Singleton<World> SWorld;