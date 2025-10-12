/*********************                                                        */
/*! \file DependencyAnalyzer.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Raya E. 
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2025 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** DependencyAnalyzer
 ** -------------------
 ** Built once per incremental batch from a base InputQuery that already has
 ** covering input bounds. It retains a non-owning pointer to that base
 ** InputQuery, and (internally, in the .cpp) constructs a preprocessed
 ** Query (which owns the NetworkLevelReasoner). The analyzer is then shared
 ** across multiple per-point solves to enable future reuse.
 **/
#ifndef __DependencyAnalyzer_h__
#define __DependencyAnalyzer_h__

#include "InputQuery.h"
#include "Query.h"
#include "NetworkLevelReasoner.h"

// #include <memory>

class DependencyAnalyzer
{
public:
    /*
      Construct the analyzer from a base InputQuery.
      NOTE: The analyzer does not take ownership; the caller must ensure
            the pointed-to InputQuery outlives this analyzer, or that the
            analyzer doesn’t dereference it after destruction of the base.
    */
    explicit DependencyAnalyzer( const InputQuery *baseIpq );

    ~DependencyAnalyzer();
    /*
      Build internal state (preprocessed query + NLR) from the base IPQ.
      Call once right after construction.
    */
    void buildFromBase();
    /*
      Accessor for the stored base InputQuery pointer (may be nullptr).
      Intended for internal/diagnostic use only.
    */
    const InputQuery *getBaseInputQuery() const;

    /*
      Lightweight diagnostic hook (safe to call anytime)
    */
    void printSummary() const;


private:
    /*
      Non-owning pointer to the base InputQuery provided by the builder
    */
    const InputQuery *_baseIpq; // non-owning, read-only pointer (MVP)

    /*
      Preprocessed Query (owns the NLR); created in the .cpp
    */
    std::unique_ptr<Query> _preprocessedQuery;

    /*
      Cached raw pointer to the NLR owned by _preprocessedQuery.
    */
    NLR::NetworkLevelReasoner *_networkLevelReasoner;

};

#endif // __DependencyAnalyzer_h__

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
