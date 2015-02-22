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
#include "linreg.hpp"

#include <random>
#include <iterator>
#include <valarray>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>

enum ScenarioType
{
    S1,
    S2,
    S3
};

typedef double real_type;

std::vector<double> do_lin_reg(
    const num::array2d<real_type> & i_X_train,
    const std::valarray<real_type> & i_y_train,
    const num::array2d<real_type> & i_X_test
)
{
    typedef num::array2d<real_type> array_type;
    typedef std::valarray<real_type> vector_type;

    // I'll be adding the intercept column
    const num::size_type NUM_FEAT{i_X_train.shape().second + 1};

    // let's map input training features onto what we'll work with
    // first column will be 1s for the intercept
    array_type X_train = num::ones<real_type>({i_X_train.shape().first, NUM_FEAT});

    // the rest will be copied from i_X_train
    // X_train[:, 1:] = i_X_train[:, :]
    X_train[X_train.columns(1, -1)] = i_X_train[i_X_train.columns(0, -1)];

    // same with test features
    array_type X_test = num::ones<real_type>({i_X_test.shape().first, NUM_FEAT});

    // X_test[:, 1:] = i_X_test[:, :]
    X_test[X_test.columns(1, -1)] = i_X_test[i_X_test.columns(0, -1)];

    vector_type y_train = i_y_train;
    vector_type theta(0.0, X_train.shape().second);

    // standardization
    for (num::size_type c{1}; c < X_train.shape().second; ++c)
    {
        const vector_type & col = X_train[X_train.column(c)];
        const vector_type & colt = X_test[X_test.column(c)];

        const real_type mu = num::mean<real_type>(col);
        const real_type dev = num::std<real_type>(col);

        X_train[X_train.column(c)] = col - mu;
        X_train[X_train.column(c)] = col / dev;

        X_test[X_test.column(c)] = colt - mu;
        X_test[X_test.column(c)] = colt / dev;
    }

    num::LinearRegression<real_type> linRegClassifier(
        num::LinearRegression<real_type>::array_type{X_train},
        num::LinearRegression<real_type>::vector_type{y_train},
        num::LinearRegression<real_type>::vector_type{theta},
        0.02,
        200
    );

    std::vector<double> result(i_X_test.shape().first);

    return result;
}

std::pair<num::array2d<real_type>, num::array2d<real_type>>
repair_X_data(
    const num::array2d<real_type> & tr_array,
    const num::array2d<real_type> & ts_array
)
{
    assert(tr_array.shape().second == ts_array.shape().second);

    typedef std::valarray<real_type> vector_type;
    typedef num::array2d<real_type> array_type;

    array_type tr_result = tr_array;
    array_type ts_result = ts_array;

    std::random_device rd;
    std::mt19937 g(rd());

    auto draw_element = [&g](const vector_type & vec) -> real_type
    {
        std::uniform_int_distribution<num::size_type> dist{0, vec.size()};

        real_type drawn;

        do
        {
            drawn = vec[dist(g)];
        } while (std::isnan(drawn));

        return drawn;
    };

    for (num::size_type cidx{0}; cidx < tr_result.shape().second; ++cidx)
    {
        vector_type column(tr_result.shape().first + ts_result.shape().first);

        column[std::slice(0, tr_result.shape().first, 1)] = tr_result[tr_result.column(cidx)];
        column[std::slice(tr_result.shape().first, ts_result.shape().first, 1)] = ts_result[ts_result.column(cidx)];

        for (num::size_type ridx{0}; ridx < tr_result.shape().first; ++ridx)
        {
            real_type & element = tr_result.at(ridx, cidx);
            if (std::isnan(element))
            {
                element = draw_element(column);
            }
        }
        for (num::size_type ridx{0}; ridx < ts_result.shape().first; ++ridx)
        {
            real_type & element = ts_result.at(ridx, cidx);
            if (std::isnan(element))
            {
                element = draw_element(column);
            }
        }
    }

    return std::make_pair(tr_result, ts_result);
}

std::valarray<real_type>
flatten_y_data(
    const num::array2d<real_type> & array,
    const std::vector<std::pair<num::size_type, num::size_type>> & subject_ranges
)
{
    typedef std::valarray<real_type> vector_type;

    vector_type result(subject_ranges.size());

    std::transform(subject_ranges.cbegin(), subject_ranges.cend(), std::begin(result),
        [&array](const std::pair<num::size_type, num::size_type> & range) -> real_type
        {
            return array.at(range.second, -1);
        }
    );

    return result;
}

num::array2d<real_type>
flatten_X_data(
    enum ScenarioType scenario,
    const num::array2d<real_type> & array,
    const std::vector<std::pair<num::size_type, num::size_type>> & subject_ranges
)
{
    typedef std::valarray<real_type> vector_type;
    typedef num::array2d<real_type> array_type;

    const std::valarray<num::size_type> s2_selector[] =
    {
        // these are column indices among those already selected from the full set
        {2, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15}, // age: 1
        {2, 6}, // age: 123
        {2, 4, 5, 6, 7, 8, 9}, // age: 366
        {2, 3, 5, 6, 7, 8, 9}, // age: 1462
        {2, 3, 5} // age: 2558
    };

    const std::valarray<num::size_type> s3_selector[] =
    {
        // these are column indices among those already selected from the full set
        {2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26}, // age: 1
        {2, 6}, // age: 123
        {2, 4, 5, 6, 7, 8, 9}, // age: 366
        {2, 3, 5, 6, 7, 8, 9}, // age: 1462
        {2, 3, 5} // age: 2558
    };

    num::shape_type shape;

    switch (scenario)
    {
        case ScenarioType::S1:
            shape = {subject_ranges.size(), 6};
            break;

        case ScenarioType::S2:
            shape =
                {
                    subject_ranges.size(),
                    std::accumulate(
                        std::begin(s2_selector),
                        std::end(s2_selector),
                        0,
                        [](const num::size_type & sum, const std::valarray<num::size_type> & selector)
                        {
                            return sum + selector.size();
                        }
                    )
                };
            break;

        case ScenarioType::S3:
            shape =
                {
                    subject_ranges.size(),
                    std::accumulate(
                        std::begin(s3_selector),
                        std::end(s3_selector),
                        0,
                        [](const num::size_type & sum, const std::valarray<num::size_type> & selector)
                        {
                            return sum + selector.size();
                        }
                    )
                };
            break;
    }

    array_type result(shape, NAN);

#if 0
    1: 3 5 6 7 8 9 10 12 14 17 18   // 11
    2: 3 7                          // 2
    3: 3 5 6 7 8 9 10               // 7
    4: 3 4 6 7 8 9 10               // 7
    5: 3 4 6                        // 3

    const std::valarray<num::size_type> s2_selector[] =
    {
        // these are column indices among those already selected from the full set
        {2, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15}, // age: 1
        {2, 6}, // age: 123
        {2, 4, 5, 6, 7, 8, 9}, // age: 366
        {2, 3, 5, 6, 7, 8, 9}, // age: 1462
        {2, 3, 5} // age: 2558
    };

//  0 1     2        3      4       5       6       7       8       9       10 11 12 13 14 15 16
//  0 1     2        3      4       5       6       7       8       9       11 13 14 15 16 17 27
//  1 2     3        4      5       6       7       8       9       10      12 14 15 16 17 18 27
    1 1     2.041    nan    47      9.23947 -2.86   -1.24   -3.65   -3.81   2 39 2041 47 8 9 nan
    1 123   5.19     nan    nan     nan     -1.75   nan     nan     nan     2 39 2041 47 8 9 nan
    1 366   9.5      nan    73      17.827  0.48    -0.41   0.88    0.97    2 39 2041 47 8 9 nan
    1 1462  16.8     100    nan     16.8    0.32    -0.64   1.06    1.02    2 39 2041 47 8 9 nan
    1 2558  23.5     121    nan     16.0508 nan     nan     nan     nan     2 39 2041 47 8 9 88

    10 1    3.685    nan    53      13.1185 1       1.97    -1.01   -0.16   2 39 3685 53 9 10 nan
    10 366  9.1      nan    71      18.052  0.13    -1.18   0.91    1.1     2 39 3685 53 9 10 nan
    10 1462 14.7     99     nan     14.9985 -0.65   -0.87   -0.18   -0.19   2 39 3685 53 9 10 nan
    10 2558 20.3     119    nan     14.3351 nan     nan     nan     nan     2 39 3685 53 9 10 101

    11 1    2.665    nan    50      10.66   -1.21   0.37    -2.64   -2.37   2 41 2665 50 4 9 nan
    11 123  4.25     nan    nan     nan     -3.33   nan     nan     nan     2 41 2665 50 4 9 nan
    11 366  7.4      nan    69      15.543  -1.58   -1.96   -0.81   -0.58   2 41 2665 50 4 9 nan
    11 1462 13.7     96     nan     14.8655 -1.18   -1.57   -0.36   -0.29   2 41 2665 50 4 9 nan
    11 2558 19.6     113    nan     15.3497 nan     nan     nan     nan     2 41 2665 50 4 9 106

    12 1    3.742    nan    55      12.3702 0.84    2.61    -2.35   -0.84   1 42 3742 55 7 9 nan
    12 123  6.12     nan    nan     nan     -1.21   nan     nan     nan     1 42 3742 55 7 9 nan
    12 366  9.6      nan    74      17.531  -0.05   -0.75   0.38    0.53    1 42 3742 55 7 9 nan
    12 1462 17.1     101    nan     16.7631 0.35    -0.56   1.06    1.07    1 42 3742 55 7 9 nan
    12 2558 23.1     121    nan     15.7776 nan     nan     nan     nan     1 42 3742 55 7 9 117

    13 1    3.147    nan    49      13.107  -0.36   -0.56   0.06    -0.23   1 40 3147 49 nan 7 nan
    13 123  6.46     nan    nan     nan     -0.74   nan     nan     nan     1 40 3147 49 nan 7 nan
    13 366  9.5      nan    79      15.2219 -0.15   1.35    -0.94   -1.25   1 40 3147 49 nan 7 nan
    13 1462 16.7     102    nan     16.0515 0.16    -0.32   0.56    0.55    1 40 3147 49 nan 7 nan
    13 2558 24.7     124    nan     16.064  nan     nan     nan     nan     1 40 3147 49 nan 7 85
////////////////////////////////////////////////////////////////////////////////
    const std::valarray<num::size_type> s3_selector[] =
    {
        // these are column indices among those already selected from the full set
        {2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26}, // age: 1
        {2, 6}, // age: 123
        {2, 4, 5, 6, 7, 8, 9}, // age: 366
        {2, 3, 5, 6, 7, 8, 9}, // age: 1462
        {2, 3, 5} // age: 2558
    };

//  0 1     2        3      4       5       6       7       8       9       10  11  12  13 14   15 16 17    18
//  1 2     3        4      5       6       7       8       9       10      11  12  13  14 15   16 17 18    19 0 1 2 3 4 5 6 7
    1 1     2.041    nan    47      9.23947 -2.86   -1.24   -3.65   -3.81   66  2   2   39 2041 47 8 9      23 1 1 7 3 4 9 4 nan
    1 123   5.19     nan    nan     nan     -1.75   nan     nan     nan     66  2   2   39 2041 47 8 9      23 1 1 7 3 4 9 4 nan
    1 366   9.5      nan    73      17.827  0.48    -0.41   0.88    0.97    66  2   2   39 2041 47 8 9      23 1 1 7 3 4 9 4 nan
    1 1462  16.8     100    nan     16.8    0.32    -0.64   1.06    1.02    66  2   2   39 2041 47 8 9      23 1 1 7 3 4 9 4 nan
    1 2558  23.5     121    nan     16.0508 nan     nan     nan     nan     66  2   2   39 2041 47 8 9      23 1 1 7 3 4 9 4 88

    2 1     2.466    nan    52      9.11982 -1.71   1.44    -4.8    -3.93   82  2   2   38 2466 52 10 10    21 1 1 0 2 2 12 3 nan
    2 123   5.44     nan    nan     nan     -1.37   nan     nan     nan     82  2   2   38 2466 52 10 10    21 1 1 0 2 2 12 3 nan
    2 366   9        nan    79      14.4208 0.04    1.92    -1.07   -1.46   82  2   2   38 2466 52 10 10    21 1 1 0 2 2 12 3 nan
    2 1462  13.7     96     nan     14.8655 -1.18   -1.57   -0.36   -0.29   82  2   2   38 2466 52 10 10    21 1 1 0 2 2 12 3 nan
    2 2558  18.6     116    nan     13.8228 nan     nan     nan     nan     82  2   2   38 2466 52 10 10    21 1 1 0 2 2 12 3 91

    12 1    3.742    nan    55      12.3702 0.84    2.61    -2.35   -0.84   5   1   1   42 3742 55 7 9      23 2 1 0 1 1 14 5 nan
    12 123  6.12     nan    nan     nan     -1.21   nan     nan     nan     5   1   1   42 3742 55 7 9      23 2 1 0 1 1 14 5 nan
    12 366  9.6      nan    74      17.531  -0.05   -0.75   0.38    0.53    5   1   1   42 3742 55 7 9      23 2 1 0 1 1 14 5 nan
    12 1462 17.1     101    nan     16.7631 0.35    -0.56   1.06    1.07    5   1   1   42 3742 55 7 9      23 2 1 0 1 1 14 5 nan
    12 2558 23.1     121    nan     15.7776 nan     nan     nan     nan     5   1   1   42 3742 55 7 9      23 2 1 0 1 1 14 5 117

    13 1    3.147    nan    49      13.107  -0.36   -0.56   0.06    -0.23   37  1   2   40 3147 49 nan 7    33 1 1 0 4 4 4 nan nan
    13 123  6.46     nan    nan     nan     -0.74   nan     nan     nan     37  1   2   40 3147 49 nan 7    33 1 1 0 4 4 4 nan nan
    13 366  9.5      nan    79      15.2219 -0.15   1.35    -0.94   -1.25   37  1   2   40 3147 49 nan 7    33 1 1 0 4 4 4 nan nan
    13 1462 16.7     102    nan     16.0515 0.16    -0.32   0.56    0.55    37  1   2   40 3147 49 nan 7    33 1 1 0 4 4 4 nan nan
    13 2558 24.7     124    nan     16.064  nan     nan     nan     nan     37  1   2   40 3147 49 nan 7    33 1 1 0 4 4 4 nan 85

    22 1    2.892    nan    49      12.045  -0.67   -0.17   -0.98   -1.07   66  2   2   38 2892 49 9 9      30 1 1 0 8 8 12 4 nan
    22 123  5.67     nan    nan     nan     -1.03   nan     nan     nan     66  2   2   38 2892 49 9 9      30 1 1 0 8 8 12 4 nan
    22 366  9.5      nan    70      19.3878 0.48    -1.57   1.63    1.88    66  2   2   38 2892 49 9 9      30 1 1 0 8 8 12 4 nan
    22 1462 15.2     98     nan     15.8267 -0.41   -1.1    0.39    0.39    66  2   2   38 2892 49 9 9      30 1 1 0 8 8 12 4 nan
    22 2558 20.4     115    nan     15.4253 nan     nan     nan     nan     66  2   2   38 2892 49 9 9      30 1 1 0 8 8 12 4 90

#endif

    switch (scenario)
    {
        case ScenarioType::S1:
            result[result.columns(0, -1)] = array[array.columns(1, 6)];
            break;

        case ScenarioType::S2:
            for (num::size_type ridx{0}; ridx < subject_ranges.size(); ++ridx)
            {
                const vector_type ages = array[std::gslice(
                    // 1st column in subject_ranges[ridx].first'th row
                    1 + subject_ranges[ridx].first * array.shape().second,
                    // # of rows per range, 1 column stride
                    {subject_ranges[ridx].second - subject_ranges[ridx].first + 1, 1},
                    // every # elements, 1 column element step
                    {array.shape().second, 1u}
                )];

                vector_type row(NAN, shape.second);
                num::size_type age_idx{0};

                auto mapper = [&ages, &row, &array, &subject_ranges, &ridx, &age_idx](
                    const real_type age,
                    const num::size_type row_idx,
                    const std::valarray<num::size_type> & selector
                ) -> num::size_type
                {
                    if (std::find(std::begin(ages), std::end(ages), age) != std::end(ages))
                    {
                        row[std::slice(row_idx, selector.size(), 1)] =
                            array[array.row(subject_ranges[ridx].first + age_idx)][selector];
                        ++age_idx;
                    }

                    return row_idx + selector.size();
                };

                num::size_type oidx{0};

                oidx = mapper(1, oidx, s2_selector[0]);
                oidx = mapper(123, oidx, s2_selector[1]);
                oidx = mapper(366, oidx, s2_selector[2]);
                oidx = mapper(1462, oidx, s2_selector[3]);
                oidx = mapper(2558, oidx, s2_selector[4]);

                assert(oidx == row.size());
                assert(age_idx == (subject_ranges[ridx].second - subject_ranges[ridx].first + 1));

                result[result.row(ridx)] = row;
//                if (ridx < 25)
//                {
//                    std::copy(std::begin(row), std::end(row), std::ostream_iterator<real_type>(std::cerr, " "));
//                    std::cerr << std::endl;
//                }
            }
            break;

        case ScenarioType::S3:
            for (num::size_type ridx{0}; ridx < subject_ranges.size(); ++ridx)
            {
                const vector_type ages = array[std::gslice(
                    // 1st column in subject_ranges[ridx].first'th row
                    1 + subject_ranges[ridx].first * array.shape().second,
                    // # of rows per range, 1 column stride
                    {subject_ranges[ridx].second - subject_ranges[ridx].first + 1, 1},
                    // every # elements, 1 column element step
                    {array.shape().second, 1u}
                )];

                vector_type row(NAN, shape.second);
                num::size_type age_idx{0};

                auto mapper = [&ages, &row, &array, &subject_ranges, &ridx, &age_idx](
                    const real_type age,
                    const num::size_type row_idx,
                    const std::valarray<num::size_type> & selector
                ) -> num::size_type
                {
                    if (std::find(std::begin(ages), std::end(ages), age) != std::end(ages))
                    {
                        row[std::slice(row_idx, selector.size(), 1)] =
                            array[array.row(subject_ranges[ridx].first + age_idx)][selector];
                        ++age_idx;
                    }

                    return row_idx + selector.size();
                };

                num::size_type oidx{0};

                oidx = mapper(1, oidx, s3_selector[0]);
                oidx = mapper(123, oidx, s3_selector[1]);
                oidx = mapper(366, oidx, s3_selector[2]);
                oidx = mapper(1462, oidx, s3_selector[3]);
                oidx = mapper(2558, oidx, s3_selector[4]);

                assert(oidx == row.size());
                assert(age_idx == (subject_ranges[ridx].second - subject_ranges[ridx].first + 1));

                result[result.row(ridx)] = row;
//                if (ridx < 25)
//                {
//                    std::copy(std::begin(row), std::end(row), std::ostream_iterator<real_type>(std::cerr, " "));
//                    std::cerr << std::endl;
//                }
            }
            break;
    }

    return result;
}

struct ChildStuntedness5
{
    enum TestType
    {
        Example,
        Provisional,
        System
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
    assert(scenario <= ScenarioType::S3);

    enum col
    {
        subjid,
        agedays,
        wtkg,
        htcm,
        lencm,
        bmi,
        waz,
        haz,
        whz,
        baz,
        siteid,
        sexn,
        feedingn,
        gagebrth,
        birthwt,
        birthlen,
        apgar1,
        apgar5,
        mage,
        demo1n,
        mmaritn,
        mcignum,
        parity,
        gravida,
        meducyrs,
        demo2n,
        geniq
    };

    typedef num::array2d<real_type> array_type;
    typedef std::valarray<real_type> vector_type;

    const std::vector<std::pair<num::size_type, num::size_type>> tr_subject_ranges =
        extract_subject_ranges(std::vector<std::string>{i_training});
    const std::vector<std::pair<num::size_type, num::size_type>> ts_subject_ranges =
        extract_subject_ranges(std::vector<std::string>{i_testing});

    auto na_xlt = [](const char * str) -> real_type
    {
        return (std::strcmp(str, "NA") == 0) ? NAN : std::strtod(str, nullptr);
    };

    const num::loadtxtCfg<real_type>::use_cols_type tr_use_cols[] =
    {
        {
            col::subjid,
            col::sexn,
            col::gagebrth, col::birthwt, col::birthlen, col::apgar1, col::apgar5,
            col::geniq
        },
        {
            col::subjid,
            col::agedays, col::wtkg, col::htcm, col::lencm, col::bmi, col::waz, col::haz, col::whz, col::baz,
            col::sexn,
            col::gagebrth, col::birthwt, col::birthlen, col::apgar1, col::apgar5,
            col::geniq
        },
        {
            col::subjid,
            col::agedays, col::wtkg, col::htcm, col::lencm, col::bmi, col::waz, col::haz, col::whz, col::baz,
            col::siteid,
            col::sexn,
            col::feedingn,
            col::gagebrth, col::birthwt, col::birthlen, col::apgar1, col::apgar5,
            col::mage, col::demo1n, col::mmaritn, col::mcignum, col::parity, col::gravida, col::meducyrs, col::demo2n,
            col::geniq
        }
    };
    const num::loadtxtCfg<real_type>::use_cols_type ts_use_cols[] =
    {
        {
            col::subjid,
            col::sexn,
            col::gagebrth, col::birthwt, col::birthlen, col::apgar1, col::apgar5
        },
        {
            col::subjid,
            col::agedays, col::wtkg, col::htcm, col::lencm, col::bmi, col::waz, col::haz, col::whz, col::baz,
            col::sexn,
            col::gagebrth, col::birthwt, col::birthlen, col::apgar1, col::apgar5
        },
        {
            col::subjid,
            col::agedays, col::wtkg, col::htcm, col::lencm, col::bmi, col::waz, col::haz, col::whz, col::baz,
            col::siteid,
            col::sexn,
            col::feedingn,
            col::gagebrth, col::birthwt, col::birthlen, col::apgar1, col::apgar5,
            col::mage, col::demo1n, col::mmaritn, col::mcignum, col::parity, col::gravida, col::meducyrs, col::demo2n
        }
    };

    array_type i_train_data =
        num::loadtxt(
            std::move(i_training),
            std::move(
                num::loadtxtCfg<real_type>()
                .delimiter(',')
                .converters({{-1, na_xlt}})
                .use_cols(tr_use_cols[scenario])
            )
        );
    std::cerr << i_train_data.shape() << std::endl;

    array_type i_test_data =
        num::loadtxt(
            std::move(i_testing),
            std::move(
                num::loadtxtCfg<real_type>()
                .delimiter(',')
                .converters({{-1, na_xlt}})
                .use_cols(ts_use_cols[scenario])
            )
        );
    std::cerr << i_test_data.shape() << std::endl;

//    for (int i = 0; i < 35; ++i)
//    {
//        for (auto v : vector_type{i_train_data[i_train_data.row(i)]})
//        {
//            std::cerr << v << " ";
//        }
//        std::cerr << std::endl;
//    }
//    std::cerr << std::endl;
//    for (int i = 0; i < 25; ++i)
//    {
//        for (auto v : vector_type{i_test_data[i_test_data.row(i)]})
//        {
//            std::cerr << v << " ";
//        }
//        std::cerr << std::endl;
//    }

    const vector_type y_tr_data = flatten_y_data(i_train_data, tr_subject_ranges);
    const enum ScenarioType enumerated_scenario = static_cast<enum ScenarioType>(scenario);

    array_type X_tr_data = flatten_X_data(enumerated_scenario, i_train_data, tr_subject_ranges);
    array_type X_ts_data = flatten_X_data(enumerated_scenario, i_test_data, ts_subject_ranges);

//    X_tr_data = repair_X_data(X_tr_data);
//    X_ts_data = repair_X_data(X_ts_data);
    auto X_tr_ts_data = repair_X_data(X_tr_data, X_ts_data);
    array_type complete_X_tr_data = std::move(X_tr_ts_data.first);
    array_type complete_X_ts_data = std::move(X_tr_ts_data.second);

    std::vector<double> result = do_lin_reg(complete_X_tr_data, y_tr_data, complete_X_ts_data);

    return result;
}

#endif /* CHILDSTUNTEDNESS5_HPP_ */
