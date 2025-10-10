/*********************                                                        */
/*! \file DependencyAnalyzer.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Raya E. (initial MVP scaffolding)
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2025 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** DependencyAnalyzer (MVP)
 ** ------------------------
 ** Minimal class that records the base InputQuery used to initialize
 ** incremental analysis. No copying of InputQuery is performed.
 ** The analyzer does not own the InputQuery; it only keeps a const pointer
 ** for reference/debug/validation.
 **/
#ifndef __DependencyAnalyzer_h__
#define __DependencyAnalyzer_h__

#include "InputQuery.h"

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
      Accessor for the stored base InputQuery pointer (may be nullptr).
      Intended for internal/diagnostic use only.
    */
    const InputQuery *getBaseInputQuery() const;

private:
    const InputQuery *_baseIpq; // non-owning, read-only pointer (MVP)
};

#endif // __DependencyAnalyzer_h__

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
