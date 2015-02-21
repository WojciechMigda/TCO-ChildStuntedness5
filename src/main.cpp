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
#include <valarray>
#include <iterator>
#include <utility>

std::vector<std::string> read_file(std::string && fname)
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
    for (auto it = subject_ranges.cbegin(); it != subject_ranges.cbegin() + PIVOT; ++it)
    {
        train_data.insert(train_data.end(), vcsv.cbegin() + it->first, vcsv.cbegin() + it->second + 1);
    }

    std::vector<std::string> test_data;
    for (auto it = subject_ranges.cbegin() + PIVOT; it != subject_ranges.cend(); ++it)
    {
        test_data.insert(test_data.end(), vcsv.cbegin() + it->first, vcsv.cbegin() + it->second + 1);
    }

    std::cerr << "Train data has " << train_data.size() << " elements" << std::endl;
    std::cerr << "Test data has " << test_data.size() << " elements" << std::endl;

    ////////////////////////////////////////////////////////////////////////////
    const ChildStuntedness5 worker;
    std::vector<double> prediction = worker.predict(0, 0, train_data, test_data);
    ////////////////////////////////////////////////////////////////////////////

    return 0;
}
