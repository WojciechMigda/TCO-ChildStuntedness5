/*******************************************************************************
 * Copyright (c) 2015 Wojciech Migda
 * All rights reserved
 * Distributed under the terms of the GNU LGPL v3
 *******************************************************************************
 *
 * Filename: extract_subject_ranges.hpp
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
 * 2015-02-22   wm              Initial version
 *
 ******************************************************************************/

#ifndef EXTRACT_SUBJECT_RANGES_HPP_
#define EXTRACT_SUBJECT_RANGES_HPP_

#include "num.hpp"

#include <vector>
#include <utility>
#include <cstdlib>
#include <cassert>

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

#endif /* EXTRACT_SUBJECT_RANGES_HPP_ */
