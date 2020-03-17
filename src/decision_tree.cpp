/*
 *    This file is under the MIT license.
 *
 *    Copyright (c) 2020 ReFantasy.cn
 *
 *    Created by tandongliang on 2020/3/16.
 *
 *    Description：
 */

#include <cmath>
#include "decision_tree.h"

double Entropy(const std::vector<double> &p)
{
	double h=0;
	for(const auto &e:p)
	{
		if(e!=0)
		    h += e * std::log2(e);
	}

	return -h;
}

