#!/usr/bin/env julia

using Pkg
ENV["JULIA_PKG_PRECOMPILE_AUTO"]=0
Pkg.add(Pkg.PackageSpec(name="JSON", version=v"0.21.4"))
Pkg.add(Pkg.PackageSpec(name="MriResearchTools", version=v"3.2.0"))
