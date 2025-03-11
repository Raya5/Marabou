/*********************                                                        */
/*! \file DependencyManager.h
 ** \verbatim
 ** Top contributors (to current version):
 **   [Your Name]
 ** This file is part of the [Your Project Name] project.
 ** Copyright (c) 2022-2025 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** DependencyManager is responsible for registering and retrieving dependencies.
 **/

#ifndef __DependencyManager_h__
#define __DependencyManager_h__

#include "IBoundManager.h"

#include <memory>
#include <unordered_map>
#include <string>
#include <iostream> // For printing

/*
  A class responsible for managing and applying dependencies in a system.
*/
class DependencyManager
{
public:
    DependencyManager( int m );
    // DependencyManager( IBoundManager &boundManager );
    // ~DependencyManager();

    // /*
    //   Get BoundManager reference
    //  */
    // IBoundManager &getBoundManager() const
    // {
    //     return _boundManager;
    // }

    // /*
    //   Registers a dependency by name.
    // */
    // template <typename T>
    // void registerDependency(const std::string& name, std::shared_ptr<T> dependency) {
    //     dependencies[name] = dependency;
    // }

    // /*
    //   Retrieves a dependency by name.
    // */
    // template <typename T>
    // std::shared_ptr<T> getDependency(const std::string& name) {
    //     if (dependencies.find(name) != dependencies.end()) {
    //         return std::static_pointer_cast<T>(dependencies[name]);
    //     }
    //     return nullptr;
    // }

    /*
      A simple function that prints a test message.
    */
    void printTestMessage() const {
        std::cout << "DependencyManager is working correctly!" << _test_var << std::endl;
    }
    void testPassBoundManager( IBoundManager *boundManager ) const {
        std::cout << "BoundManager passed! n=" << boundManager->getNumberOfVariables() << std::endl;
    }

private:
    std::unordered_map<std::string, std::shared_ptr<void>> dependencies;
    /*
       BoundManager object stores bounds of all variables.
     */
    // IBoundManager &_boundManager;

    // /*
    //    Direct pointers to _boundManager arrays to avoid multiple dereferencing.
    //  */
    // const double *_lowerBounds;
    // const double *_upperBounds;
    int _test_var;

};

#endif // __DependencyManager_h__
