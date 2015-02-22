/*******************************************************************************
 * Copyright (c) 2015 Wojciech Migda
 * All rights reserved
 * Distributed under the terms of the GNU LGPL v3
 *******************************************************************************
 *
 * Filename: main.cpp
 *
 * Description:
 *      Child Stuntedness 5
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

#include "ChildStuntedness5.hpp"
#include "num.hpp"

#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <cstddef>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <cassert>
#include <iterator>
#include <utility>
#include <cstring>
#include <functional>

std::vector<std::string>
read_file(std::string && fname)
{
    std::ifstream fcsv(fname);
    std::vector<std::string> vcsv;

    for (std::string line; std::getline(fcsv, line); /* nop */)
    {
        vcsv.push_back(line);
    }
    fcsv.close();

    return vcsv;
}

std::vector<std::pair<num::size_type, num::size_type>>
extract_subject_ranges(std::vector<std::string> && vstr)
{
    assert(vstr.size() > 1);

    std::vector<std::pair<num::size_type, num::size_type>> result;

    num::size_type first{0};
    int curr_id = std::atoi(vstr[first].c_str());

    for (num::size_type idx{1}; idx < vstr.size(); ++idx)
    {
        const int id = std::atoi(vstr[idx].c_str());

        if (id != curr_id)
        {
            result.emplace_back(first, idx - 1);
            first = idx;
            curr_id = id;
        }
        else
        {
            ;
        }
    }
    result.emplace_back(first, vstr.size() - 1);

    return result;
}

int main(int argc, char **argv)
{
    const int SEED = (argc == 2 ? std::atoi(argv[1]) : 1);
    const char * FNAME = (argc == 3 ? argv[2] : "../data/exampleData.csv");

    std::cerr << "SEED: " << SEED << ", CSV: " << FNAME << std::endl;

    const std::vector<std::string> vcsv = read_file(std::string(FNAME));

    std::cerr << "Read " << vcsv.size() << " lines" << std::endl;

    std::vector<std::pair<num::size_type, num::size_type>> subject_ranges =
        extract_subject_ranges(std::vector<std::string>{vcsv});

    std::mt19937 g(SEED);
    std::shuffle(subject_ranges.begin(), subject_ranges.end(), g);

    const std::size_t PIVOT = 0.67 * subject_ranges.size();

    std::vector<std::string> train_data;
    std::vector<std::string> train_data0;
    for (auto it = subject_ranges.cbegin(); it != subject_ranges.cbegin() + PIVOT; ++it)
    {
        train_data.insert(train_data.end(), vcsv.cbegin() + it->first, vcsv.cbegin() + it->second + 1);
        train_data0.push_back(vcsv[it->second]);
    }

    std::vector<std::string> test_data;
    std::vector<std::string> test_data0;
    for (auto it = subject_ranges.cbegin() + PIVOT; it != subject_ranges.cend(); ++it)
    {
        test_data.insert(test_data.end(), vcsv.cbegin() + it->first, vcsv.cbegin() + it->second + 1);
        test_data0.push_back(vcsv[it->second]);
    }

    std::cerr << "Train data has " << PIVOT << " IDs" << std::endl;
    std::cerr << "Train data has " << train_data.size() << " rows" << std::endl;
    std::cerr << "Train data 0 has " << train_data0.size() << " rows" << std::endl;
    std::cerr << "Test data has " << subject_ranges.size() - PIVOT << " IDs" << std::endl;
    std::cerr << "Test data has " << test_data.size() << " rows" << std::endl;
    std::cerr << "Test data 0 has " << test_data0.size() << " rows" << std::endl;

    assert(train_data.size() + test_data.size() == vcsv.size());
    assert(train_data0.size() + test_data0.size() == subject_ranges.size());

    for (auto item : test_data)
    {
        constexpr num::size_type IQ_COL{26};

        std::size_t nth_comma{0};

        item.resize(std::distance(item.cbegin(), std::find_if(item.cbegin(), item.cend(),
            [&nth_comma](const char & ch)
            {
                if (ch == ',' && nth_comma == (IQ_COL - 1))
                {
                    return true;
                }
                else
                {
                    nth_comma += (ch == ',');
                    return false;
                }
            }
        )));
        assert(std::count(item.cbegin(), item.cend(), ',') == (IQ_COL - 1));
    }

    std::vector<double> test_iqs;

    for (auto item : test_data0)
    {
        constexpr num::size_type IQ_COL{26};

        std::size_t nth_comma{0};

        test_iqs.push_back(std::atoi(std::strrchr(item.c_str(), ',') + 1));

        item.resize(std::distance(item.cbegin(), std::find_if(item.cbegin(), item.cend(),
            [&nth_comma](const char & ch)
            {
                if (ch == ',' && nth_comma == (IQ_COL - 1))
                {
                    return true;
                }
                else
                {
                    nth_comma += (ch == ',');
                    return false;
                }
            }
        )));
        assert(std::count(item.cbegin(), item.cend(), ',') == (IQ_COL - 1));
    }

    const double MEAN_TRAIN_IQ = std::accumulate(train_data0.cbegin(), train_data0.cend(), 0.0,
        [](const double & sum, const std::string & item) -> double
        {
            return sum + std::atoi(std::strrchr(item.c_str(), ',') + 1);
        }
    ) / PIVOT;

    const double SSE0 = std::accumulate(test_iqs.cbegin(), test_iqs.cend(), 0.0,
        [&MEAN_TRAIN_IQ](const double & sse, const double & iq) -> double
        {
            return sse + (iq - MEAN_TRAIN_IQ) * (iq - MEAN_TRAIN_IQ);
        }
    );

    ////////////////////////////////////////////////////////////////////////////

    const ChildStuntedness5 worker;

    auto sse_lambda = [](const double & lhs, const double & rhs) -> double
    {
        return (lhs - rhs) * (lhs - rhs);
    };

    std::vector<double> prediction1 = worker.predict(
        ChildStuntedness5::TestType::Example,
        ChildStuntedness5::ScenarioType::S1,
        train_data0,
        test_data0);
    assert(prediction1.size() == test_iqs.size());
    const double SSE1 = std::inner_product(
        prediction1.cbegin(),
        prediction1.cend(),
        test_iqs.cbegin(),
        0.0,
        std::plus<double>(),
        sse_lambda);

//    std::vector<double> prediction2 = worker.predict(
//        ChildStuntedness5::TestType::Example,
//        ChildStuntedness5::ScenarioType::S2,
//        train_data,
//        test_data);
//    assert(prediction2.size() == test_iqs.size());
//    const double SSE2 = std::inner_product(
//        prediction2.cbegin(),
//        prediction2.cend(),
//        test_iqs.cbegin(),
//        0.0,
//        std::plus<double>(),
//        sse_lambda);
//
//    std::vector<double> prediction3 = worker.predict(
//        ChildStuntedness5::TestType::Example,
//        ChildStuntedness5::ScenarioType::S3,
//        train_data,
//        test_data);
//    assert(prediction3.size() == test_iqs.size());
//    const double SSE3 = std::inner_product(
//        prediction3.cbegin(),
//        prediction3.cend(),
//        test_iqs.cbegin(),
//        0.0,
//        std::plus<double>(),
//        sse_lambda);

    auto score_lambda = [](const double SSE, const double SSE0) -> double
    {
        return 1e6 * std::max(0.0, 1.0 - SSE / SSE0);
    };

    std::cerr << "Score 1: " << score_lambda(SSE1, SSE0) << std::endl;
//    std::cerr << "Score 2: " << score_lambda(SSE2, SSE0) << std::endl;
//    std::cerr << "Score 3: " << score_lambda(SSE3, SSE0) << std::endl;

    return 0;
}
