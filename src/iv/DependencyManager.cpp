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



DependencyManager::DependencyManager( const IBoundManager &boundManager)
    : _boundManager( boundManager )
    , _networkLevelReasoner( NULL )
    , _dependencyMaxSize(36)
{
}
DependencyManager::~DependencyManager()
{
}

void DependencyManager::logCurrentBounds() const {

    if (!_networkLevelReasoner) return;
    
    printf("==== Layer-wise Bounds ====\n");
    for (unsigned layer = 0; layer < _networkLevelReasoner->getNumberOfLayers(); ++layer) {
        printf("-- Layer %u --\n", layer);
        int stop_count = 3;
        // for (unsigned neuron : _networkLevelReasoner->getNeuronsInLayer(layer)) {
        //     double lb = _boundManager.getLowerBound(neuron);
        //     double ub = _boundManager.getUpperBound(neuron);
        //     printf("Neuron %u: [%.6f, %.6f]\n", neuron, lb, ub);
        //     stop_count--;

        //     if ( stop_count == 0)
        //         break;
        // }
    }
}

