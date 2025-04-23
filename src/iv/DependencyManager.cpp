/*********************                                                        */
/*! \file DependencyManager.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   [Your Name]
 ** This file is part of the [Your Project Name] project.
 ** Copyright (c) 2022-2025 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** Implementation of the DependencyManager class.
 **/

#include "DependencyManager.h"



DependencyManager::DependencyManager( const IBoundManager &boundManager )
    : _boundManager( boundManager )
    , _dependencyMaxSize(33)
//     // , _lowerBounds( _boundManager.getLowerBounds() )
//     // , _upperBounds( _boundManager.getUpperBounds() )
{
}
DependencyManager::~DependencyManager()
{
}

void DependencyManager::logCurrentBounds() const {
    printf("==== Current Bounds After Split ====\n");
    for (unsigned i = 0; i < 5; ++i) { //_boundManager.getNumberOfVariables()
        double lb = _boundManager.getLowerBound(i);
        double ub = _boundManager.getUpperBound(i);
        printf("Neuron %u: [%.6f, %.6f]\n", i, lb, ub);  // %.6f for precision
    }
}