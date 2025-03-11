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

// DependencyManager::DependencyManager( )//( IBoundManager &boundManager )
//     : //_boundManager( boundManager )
//     // , _lowerBounds( _boundManager.getLowerBounds() )
//     // , _upperBounds( _boundManager.getUpperBounds() )
// {
// }
// DependencyManager::~DependencyManager()
// {
// }

DependencyManager::DependencyManager( int m )
: _test_var( m )
{
    _test_var = 54;
}