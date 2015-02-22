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

#include "array2d.hpp"
#include "extract_subject_ranges.hpp"

#include <valarray>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>

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

    std::vector<double>
    predict(
        int testType,
        int scenario,
        std::vector<std::string> && training,
        std::vector<std::string> && testing) const;
};

std::vector<double>
ChildStuntedness5::predict(
    int testType,
    int scenario,
    std::vector<std::string> & training,
    std::vector<std::string> & testing) const
{
    return predict(testType, scenario, std::move(training), std::move(testing));
}

std::vector<double>
ChildStuntedness5::predict(
    int testType,
    int scenario,
    std::vector<std::string> && i_training,
    std::vector<std::string> && i_testing) const
{
    typedef num::array2d<real_type> array_type;
    typedef std::valarray<real_type> vector_type;

    const std::vector<std::pair<num::size_type, num::size_type>> tr_subject_ranges =
        extract_subject_ranges(std::vector<std::string>{i_training});
    const std::vector<std::pair<num::size_type, num::size_type>> ts_subject_ranges =
        extract_subject_ranges(std::vector<std::string>{i_testing});

    std::vector<double> result(ts_subject_ranges.size());

    auto na_xlt = [](const char * str) -> real_type
    {
        return (std::strcmp(str, "NA") == 0) ? NAN : std::strtod(str, nullptr);
    };

    array_type train_data =
        num::loadtxt(
            std::move(i_training),
            std::move(
                num::loadtxtCfg<real_type>()
                .delimiter(',')
                .converters({{-1, na_xlt}})
//                .use_cols({1, 2})
            )
        );
    std::cerr << train_data.shape() << std::endl;

    return result;
}

#endif /* CHILDSTUNTEDNESS5_HPP_ */
