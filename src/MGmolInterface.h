// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef MGMOLINTERFACE_H
#define MGMOLINTERFACE_H

#include <cstring>
#include <vector>

class MGmolInterface
{
public:
    MGmolInterface() { }

    virtual ~MGmolInterface() { }

    virtual int setupFromInput(const std::string input_file)            = 0;
    virtual int setupLRs(const std::string input_file)                  = 0;
    virtual int setupConstraintsFromInput(const std::string input_file) = 0;
    virtual void setup()                                                = 0;
    virtual void run()                                                  = 0;
    virtual void cleanup()                                              = 0;

    virtual double evaluateEnergyAndForces(const std::vector<double>& tau,
        const std::vector<short>& atnumbers, std::vector<double>& forces)
        = 0;
    virtual double evaluateEnergyAndForces(Orbitals*,
        const std::vector<double>& tau, const std::vector<short>& atnumbers,
        std::vector<double>& forces)
        = 0;
    virtual double evaluateDMandEnergyAndForces(Orbitals*,
        const std::vector<double>& tau, const std::vector<short>& atnumbers,
        std::vector<double>& forces)
        = 0;

    virtual void getAtomicPositions(std::vector<double>& tau) = 0;
    virtual void getAtomicNumbers(std::vector<short>& an)     = 0;
    virtual void setPositions(const std::vector<double>& positions,
        const std::vector<short>& atnumbers)
        = 0;
    virtual std::shared_ptr<ProjectedMatricesInterface> getProjectedMatrices()
        = 0;
    virtual void dumpRestart() = 0;
};

#endif
