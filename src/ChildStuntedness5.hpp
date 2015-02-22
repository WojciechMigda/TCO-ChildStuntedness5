/*******************************************************************************
 * Copyright (c) 2015 Wojciech Migda
 * All rights reserved
 * Distributed under the terms of the GNU LGPL v3
 *******************************************************************************
 *
 * Filename: ChildStuntedness5.hpp
 *
 * Description:
 *      description
 *
 * Authors:
 *          Wojciech Migda (wm)
 *
 *******************************************************************************
 * History:
 * --------
 * Date         Who  Ticket     Description
 * ----------   ---  ---------  ------------------------------------------------
 * 2015-02-21   wm              Initial version
 *
 ******************************************************************************/

#ifndef CHILDSTUNTEDNESS5_HPP_
#define CHILDSTUNTEDNESS5_HPP_

#include <vector>
#include <string>

typedef double real_type;

struct ChildStuntedness5
{
    enum TestType
    {
        Example,
        Provisional,
        System
    };
    enum ScenarioType
    {
        S1,
        S2,
        S3
    };

    std::vector<double>
    predict(
        int testType,
        int scenario,
        std::vector<std::string> & training,
        std::vector<std::string> & testing) const;
};

std::vector<double>
ChildStuntedness5::predict(
    int testType,
    int scenario,
    std::vector<std::string> & training,
    std::vector<std::string> & testing) const
{
    std::vector<double> result(testing.size());

    return result;
}

#endif /* CHILDSTUNTEDNESS5_HPP_ */
