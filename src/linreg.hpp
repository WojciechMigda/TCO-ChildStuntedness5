/*******************************************************************************
 * Copyright (c) 2015 Wojciech Migda
 * All rights reserved
 * Distributed under the terms of the GNU LGPL v3
 *******************************************************************************
 *
 * Filename: linreg.hpp
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

#ifndef LINREG_HPP_
#define LINREG_HPP_

#include "array2d.hpp"
#include "num.hpp"

#include <valarray>

namespace num
{

template<typename _ValueType>
class LogisticRegression
{
public:
    typedef _ValueType value_type;
    typedef std::valarray<value_type> vector_type;
    typedef array2d<value_type> array_type;

    LogisticRegression(
        array_type && X,
        vector_type && y,
        vector_type && theta0,
        value_type C,
        size_type max_iter
    );

    vector_type
    fit(void) const;

    vector_type
    predict(const array_type & X, const vector_type & theta, bool round = true) const;

    vector_type
    predict(array_type && X, vector_type && theta, bool round = true) const;

private:
    const array_type m_X;
    const vector_type m_y;
    const vector_type m_theta0;
    const value_type m_C;
    const size_type m_max_iter;
};

} // namespace num

#endif /* LINREG_HPP_ */
