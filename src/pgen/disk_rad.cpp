//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk.cpp
//! \brief Initializes stratified Keplerian accretion disk in both cylindrical and
//! spherical polar coordinates.  Initial conditions are in vertical hydrostatic eqm.

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt
#include <cstdlib>    // srand
#include <cstring>    // strcmp()
#include <fstream>
#include <iostream>   // endl
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"
#include "../nr_radiation/radiation.hpp"
#include "../nr_radiation/integrators/rad_integrators.hpp"

namespace {
void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k);
Real DenProfileCyl(const Real rad, const Real phi, const Real z);
Real PoverR(const Real rad, const Real phi, const Real z);
Real VelProfileCyl(const Real rad, const Real phi, const Real z);
// problem parameters which are useful to make global to this file
Real gm0, r0, rho0, tem0, dslope, p0_over_r0, pslope, gamma_gas, nu_iso, alpha, dens0, l0, opacity, rstar, tstar;
Real dfloor, tceil;
Real Omega0;
} // namespace

void Viscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
            int is, int ie, int js, int je, int ks, int ke);

void Source(MeshBlock *pmb, const Real time, const Real dt,
             const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
             const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
             AthenaArray<Real> &cons_scalar);

// User-defined boundary conditions for disk simulations
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskRadInnerX1(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
                    const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
                    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void DiskRadOuterX1(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
                    const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
                    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void DiskRadInnerX2(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
                    const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
                    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void DiskRadOuterX2(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
                    const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
                    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);


//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Get parameters for gravitatonal potential of central point mass
  gm0 = pin->GetOrAddReal("problem","gm0",1.0);
  r0 = pin->GetOrAddReal("problem","r0",1.0);

  // Get parameters for initial density and velocity
  rho0 = pin->GetReal("problem","rho0");
  dslope = pin->GetOrAddReal("problem","dslope",0.0);

  // Get parameters of initial pressure and cooling parameters
  if (NON_BAROTROPIC_EOS) {
    p0_over_r0 = pin->GetOrAddReal("problem","p0_over_r0",0.0025);
    pslope = pin->GetOrAddReal("problem","pslope",0.0);
    gamma_gas = pin->GetReal("hydro","gamma");
  } else {
    p0_over_r0=SQR(pin->GetReal("hydro","iso_sound_speed"));
  }
  Real float_min = std::numeric_limits<float>::min();
  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(float_min)));

  Omega0 = pin->GetOrAddReal("orbital_advection","Omega0",0.0);

  alpha = pin -> GetOrAddReal("problem", "alpha", 0.0);
  nu_iso = pin -> GetOrAddReal("problem", "nu_iso", 0.0);

  l0 = pin->GetReal("radiation", "length_unit");
  dens0 = pin->GetReal("radiation", "density_unit");
  tem0 = pin->GetReal("radiation", "T_unit");
  opacity = pin->GetOrAddReal("problem", "opacity", 0.0);

  opacity *= dens0*l0;

  rstar = pin->GetOrAddReal("problem", "rstar", 0.0);
  tstar = pin->GetOrAddReal("problem", "tstar", 0.0);

  rstar = rstar / l0;
  tstar = tstar / tem0;

  tceil = pin->GetOrAddReal("problem","tceil",1e5);
  tceil = tceil / tem0;

  if (nu_iso > 0.0) {
    EnrollViscosityCoefficient(Viscosity);
  }

  EnrollUserExplicitSourceFunction(Source);

  // enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
    if (NR_RADIATION_ENABLED || IM_RADIATION_ENABLED)
      EnrollUserRadBoundaryFunction(BoundaryFace::inner_x1, DiskRadInnerX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
    if (NR_RADIATION_ENABLED || IM_RADIATION_ENABLED)
      EnrollUserRadBoundaryFunction(BoundaryFace::outer_x1, DiskRadOuterX1);
  }
  if (mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, DiskInnerX2);
    if (NR_RADIATION_ENABLED || IM_RADIATION_ENABLED)
      EnrollUserRadBoundaryFunction(BoundaryFace::inner_x2, DiskRadInnerX2);
  }
  if (mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, DiskOuterX2);
    if (NR_RADIATION_ENABLED || IM_RADIATION_ENABLED)
      EnrollUserRadBoundaryFunction(BoundaryFace::outer_x2, DiskRadOuterX2);
  }

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real den, vel;
  Real x1, x2, x3;

  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;
  //  Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
    x3 = pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      x2 = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {
        x1 = pcoord->x1v(i);
        GetCylCoord(pcoord,rad,phi,z,i,j,k); // convert to cylindrical coordinates
        // compute initial conditions in cylindrical coordinates
        den = DenProfileCyl(rad,phi,z);
        vel = VelProfileCyl(rad,phi,z);
        if (porb->orbital_advection_defined)
          vel -= vK(porb, x1, x2, x3);
        phydro->u(IDN,k,j,i) = den;
        phydro->u(IM1,k,j,i) = 0.0;
        if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          phydro->u(IM2,k,j,i) = den*vel;
          phydro->u(IM3,k,j,i) = 0.0;
        } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = den*vel;
        }

        if (NON_BAROTROPIC_EOS) {
          Real p_over_r = PoverR(rad,phi,z);
          phydro->u(IEN,k,j,i) = p_over_r*phydro->u(IDN,k,j,i)/(gamma_gas - 1.0);
          phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))
                                       + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
        }
      }
    }
  }
  if (NR_RADIATION_ENABLED || IM_RADIATION_ENABLED) {
    int nfreq = pnrrad->nfreq;
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          for (int ifr=0; ifr < nfreq; ++ifr) {
            pnrrad->sigma_s(k,j,i,ifr) = 0.0;
            pnrrad->sigma_a(k,j,i,ifr) = opacity*phydro->u(IDN,k,j,i);
            pnrrad->sigma_pe(k,j,i,ifr) = opacity*phydro->u(IDN,k,j,i);
            pnrrad->sigma_p(k,j,i,ifr) = opacity*phydro->u(IDN,k,j,i);
          }
          for (int n=0; n<pnrrad->n_fre_ang; ++n) {
            GetCylCoord(pcoord,rad,phi,z,i,j,k);
            Real p_over_r = PoverR(rad,phi,z);
            pnrrad->ir(k,j,i,n) = std::pow(p_over_r, 4);
          }
        }
      }
    }
  }

  return;
}

// user defined outputs
//void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
//{
//  AllocateUserOutputVariables(5);
//  AllocateRealUserMeshBlockDataField(5);
//  ruser_meshblock_data[0].NewAthenaArray(block_size.nx1, block_size.nx2+2*NGHOST, block_size.nx3+2*NGHOST);
//  ruser_meshblock_data[1].NewAthenaArray(block_size.nx3, block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST);
//  ruser_meshblock_data[2].NewAthenaArray(block_size.nx3, block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST);
//  ruser_meshblock_data[3].NewAthenaArray(block_size.nx3, block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST);
//  ruser_meshblock_data[4].NewAthenaArray(block_size.nx3, block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST);
//  return;
//}
//
//void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin)
//    {
//      for(int k=ks; k<=ke; k++) {
//        for(int j=js; j<=je; j++) {
//          for(int i=is; i<=ie; i++) {
//            user_out_var(0,k,j,i) = ruser_meshblock_data[0](k,j,i);
//            user_out_var(1,k,j,i) = ruser_meshblock_data[1](k,j,i);
//            user_out_var(2,k,j,i) = ruser_meshblock_data[2](k,j,i);
//            user_out_var(3,k,j,i) = ruser_meshblock_data[3](k,j,i);
//            user_out_var(4,k,j,i) = ruser_meshblock_data[4](k,j,i);
//          }
//        }
//      }
//    }

void Viscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
               int is, int ie, int js, int je, int ks, int ke) {
    Real rad, phi, z;
    Real inv_GMroot = 1.0/std::sqrt(gm0);
    if (phdif->nu_iso > 0.0) {
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          for (int i = is; i <= ie; ++i) {
            GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
            const Real &gas_pre = prim(IPR, k, j, i);
            const Real &gas_rho = prim(IDN, k, j, i);
            Real gamma          = pmb->peos->GetGamma();
            Real inv_Omega_Keplerian = inv_GMroot*pow(rad, 1.5);
            phdif->nu(HydroDiffusion::DiffProcess::iso, k, j, i) =
            alpha * (gamma*gas_pre/gas_rho) * inv_Omega_Keplerian;
        }
      }
    }
  }
}

void Source(MeshBlock *pmb, const Real time, const Real dt,
             const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
             const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
             AthenaArray<Real> &cons_scalar)
{
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        const Real &gas_pre = prim(IPR, k, j, i);
        const Real &gas_rho = prim(IDN, k, j, i);
        Real gamma          = pmb->peos->GetGamma();
        Real gas_tem        = gas_pre / gas_rho;
        if (gas_tem > tceil) {
          cons(IEN,k,j,i) -= prim(IDN,k,j,i) * (gas_tem - tceil) / (gamma - 1.0);
        }
      }
    }
  }
  if (NR_RADIATION_ENABLED || IM_RADIATION_ENABLED) {
    int nfreq = pmb->pnrrad->nfreq;
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          for (int ifr=0; ifr < nfreq; ++ifr) {
            pmb->pnrrad->sigma_s(k,j,i,ifr) = 0.0;
            pmb->pnrrad->sigma_a(k,j,i,ifr) = opacity*pmb->phydro->u(IDN,k,j,i);
            pmb->pnrrad->sigma_pe(k,j,i,ifr) = opacity*pmb->phydro->u(IDN,k,j,i);
            pmb->pnrrad->sigma_p(k,j,i,ifr) = opacity*pmb->phydro->u(IDN,k,j,i);
          }
        }
      }
    }
  }
  return;
}

namespace {
//----------------------------------------------------------------------------------------
//! transform to cylindrical coordinate

void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k) {
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    rad=pco->x1v(i);
    phi=pco->x2v(j);
    z=pco->x3v(k);
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    rad=std::abs(pco->x1v(i)*std::sin(pco->x2v(j)));
    phi=pco->x3v(k);
    z=pco->x1v(i)*std::cos(pco->x2v(j));
  }
  return;
}

//----------------------------------------------------------------------------------------
//! computes density in cylindrical coordinates

Real DenProfileCyl(const Real rad, const Real phi, const Real z) {
  Real den;
  Real p_over_r = p0_over_r0;
  if (NON_BAROTROPIC_EOS) p_over_r = PoverR(rad, phi, z);
  Real denmid = rho0*std::pow(rad/r0,dslope);
  Real dentem = denmid*std::exp(gm0/p_over_r*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den = dentem;
  return std::max(den,dfloor);
}

//----------------------------------------------------------------------------------------
//! computes pressure/density in cylindrical coordinates

Real PoverR(const Real rad, const Real phi, const Real z) {
  Real poverr;
  poverr = p0_over_r0*std::pow(rad/r0, pslope);
  return poverr;
}

//----------------------------------------------------------------------------------------
//! computes rotational velocity in cylindrical coordinates

Real VelProfileCyl(const Real rad, const Real phi, const Real z) {
  Real p_over_r = PoverR(rad, phi, z);
  Real vel = (dslope+pslope)*p_over_r/(gm0/rad) + (1.0+pslope)
             - pslope*rad/std::sqrt(rad*rad+z*z);
  vel = std::sqrt(gm0/rad)*std::sqrt(vel) - rad*Omega0;
  return vel;
}
} // namespace

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          GetCylCoord(pco,rad,phi,z,il-i,j,k);
          prim(IDN,k,j,il-i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));
          prim(IM1,k,j,il-i) = 0.0;
          prim(IM2,k,j,il-i) = vel;
          prim(IM3,k,j,il-i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,j,il-i) = PoverR(rad, phi, z)*prim(IDN,k,j,il-i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          prim(IDN,k,j,il-i) = prim(IDN,k,j,il);
          prim(IM1,k,j,il-i) = std::min(prim(IM1,k,j,il), 0.0);
          prim(IM2,k,j,il-i) = prim(IM2,k,j,il);
          prim(IM3,k,j,il-i) = prim(IM3,k,j,il)*sqrt(pco->x1v(il)/pco->x1v(il-i));
          prim(IEN,k,j,il-i) = prim(IEN,k,j,il);
        }
      }
    }
  }
  // } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
  //   for (int k=kl; k<=ku; ++k) {
  //     for (int j=jl; j<=ju; ++j) {
  //       for (int i=1; i<=ngh; ++i) {
  //         GetCylCoord(pco,rad,phi,z,il-i,j,k);
  //         prim(IDN,k,j,il-i) = DenProfileCyl(rad,phi,z);
  //         vel = VelProfileCyl(rad,phi,z);
  //         if (pmb->porb->orbital_advection_defined)
  //           vel -= vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));
  //         std::cout << "IM1_boundary: " << prim(IM1,k,j,il) << std::endl;
  //         prim(IM1,k,j,il-i) = 0.0;
  //         prim(IM2,k,j,il-i) = 0.0;
  //         prim(IM3,k,j,il-i) = vel;
  //         if (NON_BAROTROPIC_EOS)
  //           prim(IEN,k,j,il-i) = PoverR(rad, phi, z)*prim(IDN,k,j,il-i);
  //       }
  //     }
  //   }
  // }

}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          GetCylCoord(pco,rad,phi,z,iu+i,j,k);
          prim(IDN,k,j,iu+i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));
          prim(IM1,k,j,iu+i) = 0.0;
          prim(IM2,k,j,iu+i) = vel;
          prim(IM3,k,j,iu+i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,j,iu+i) = PoverR(rad, phi, z)*prim(IDN,k,j,iu+i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          prim(IDN,k,j,iu+i) = prim(IDN,k,j,iu);
          prim(IM1,k,j,iu+i) = std::max(prim(IM1,k,j,iu), 0.0);
          prim(IM2,k,j,iu+i) = prim(IM2,k,j,iu);
          prim(IM3,k,j,iu+i) = prim(IM3,k,j,iu)*sqrt(pco->x1v(iu)/pco->x1v(iu+i));
          prim(IEN,k,j,iu+i) = prim(IEN,k,j,iu);
        }
      }
    }
  }
  // } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
  //   for (int k=kl; k<=ku; ++k) {
  //     for (int j=jl; j<=ju; ++j) {
  //       for (int i=1; i<=ngh; ++i) {
  //         GetCylCoord(pco,rad,phi,z,iu+i,j,k);
  //         prim(IDN,k,j,iu+i) = DenProfileCyl(rad,phi,z);
  //         vel = VelProfileCyl(rad,phi,z);
  //         if (pmb->porb->orbital_advection_defined)
  //           vel -= vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));
  //         prim(IM1,k,j,iu+i) = 0.0;
  //         prim(IM2,k,j,iu+i) = 0.0;
  //         prim(IM3,k,j,iu+i) = vel;
  //         if (NON_BAROTROPIC_EOS)
  //           prim(IEN,k,j,iu+i) = PoverR(rad, phi, z)*prim(IDN,k,j,iu+i);
  //       }
  //     }
  //   }
  // }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskInnerX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,jl-j,k);
          prim(IDN,k,jl-j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(jl-j), pco->x3v(k));
          prim(IM1,k,jl-j,i) = 0.0;
          prim(IM2,k,jl-j,i) = vel;
          prim(IM3,k,jl-j,i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,jl-j,i) = PoverR(rad, phi, z)*prim(IDN,k,jl-j,i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          prim(IDN,k,jl-j,i) = prim(IDN,k,jl,i);
          prim(IM1,k,jl-j,i) = prim(IM1,k,jl,i);
          prim(IM2,k,jl-j,i) = std::min(prim(IM2,k,jl,i), 0.0);
          prim(IM3,k,jl-j,i) = prim(IM3,k,jl,i);
          prim(IEN,k,jl-j,i) = prim(IEN,k,jl,i);
        }
      }
    }
  }
  // } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
  //   for (int k=kl; k<=ku; ++k) {
  //     for (int j=1; j<=ngh; ++j) {
  //       for (int i=il; i<=iu; ++i) {
  //         GetCylCoord(pco,rad,phi,z,i,jl-j,k);
  //         prim(IDN,k,jl-j,i) = DenProfileCyl(rad,phi,z);
  //         vel = VelProfileCyl(rad,phi,z);
  //         if (pmb->porb->orbital_advection_defined)
  //           vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(jl-j), pco->x3v(k));
  //         prim(IM1,k,jl-j,i) = 0.0;
  //         prim(IM2,k,jl-j,i) = 0.0;
  //         prim(IM3,k,jl-j,i) = vel;
  //         if (NON_BAROTROPIC_EOS)
  //           prim(IEN,k,jl-j,i) = PoverR(rad, phi, z)*prim(IDN,k,jl-j,i);
  //       }
  //     }
  //   }
  // }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskOuterX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,ju+j,k);
          prim(IDN,k,ju+j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(ju+j), pco->x3v(k));
          prim(IM1,k,ju+j,i) = 0.0;
          prim(IM2,k,ju+j,i) = vel;
          prim(IM3,k,ju+j,i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,ju+j,i) = PoverR(rad, phi, z)*prim(IDN,k,ju+j,i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          prim(IDN,k,ju+j,i) = prim(IDN,k,ju,i);
          prim(IM1,k,ju+j,i) = prim(IM1,k,ju,i);
          prim(IM2,k,ju+j,i) = std::max(prim(IM2,k,ju,i), 0.0);
          prim(IM3,k,ju+j,i) = prim(IM3,k,ju,i);
          prim(IEN,k,ju+j,i) = prim(IEN,k,ju,i);
        }
      }
    }
  }
  // } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
  //   for (int k=kl; k<=ku; ++k) {
  //     for (int j=1; j<=ngh; ++j) {
  //       for (int i=il; i<=iu; ++i) {
  //         GetCylCoord(pco,rad,phi,z,i,ju+j,k);
  //         prim(IDN,k,ju+j,i) = DenProfileCyl(rad,phi,z);
  //         vel = VelProfileCyl(rad,phi,z);
  //         if (pmb->porb->orbital_advection_defined)
  //           vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(ju+j), pco->x3v(k));
  //         prim(IM1,k,ju+j,i) = 0.0;
  //         prim(IM2,k,ju+j,i) = 0.0;
  //         prim(IM3,k,ju+j,i) = vel;
  //         if (NON_BAROTROPIC_EOS)
  //           prim(IEN,k,ju+j,i) = PoverR(rad, phi, z)*prim(IDN,k,ju+j,i);
  //       }
  //     }
  //   }
  // }
}

void DiskRadInnerX1(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
              const AthenaArray<Real> &w, FaceField &b,
              AthenaArray<Real> &ir,
              Real time, Real dt,
              int is, int ie, int js, int je, int ks, int ke, int ngh) {
  Real sum_weight;          // sum of weights and miu_x throughout all angles
  Real miux_min = 0;        // critical angle 
  Real flux = 0;            // flux at the inner boundary
  int nang=prad->nang;
  int nfreq=prad->nfreq;

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        for (int ifr=0; ifr<nfreq; ++ifr) {
          sum_weight = 0;
          miux_min = sqrt(pco->x1v(is-i)*pco->x1v(is-i) - rstar*rstar)/pco->x1v(is-i);

          for(int n=0; n<nang; ++n){
            int ang = ifr*nang + n;
            if (prad->mu(0,k,j,is-i,ang) > miux_min){
              sum_weight += prad->wmu(n) * prad->mu(0,k,j,is-i,ang);
            } else {
              sum_weight += 0.0;
            }
          }

          for(int n=0; n<nang; ++n){
            int ang = ifr*nang + n;
            flux = std::pow(tstar, 4)*std::pow(rstar/pco->x1v(is-i), 2) / 4;
            ir(k,j,is-i,ang) = 0.0;
            if (nfreq == 1) {
              flux *= 1.0;
              // std::cout << "i: " << i << "flux: " << flux << std::endl;
            } else {
              if (ifr < nfreq-1) {
                flux *= prad->IntPlanckFunc(prad->nu_grid(ifr)/tstar,
                                        prad->nu_grid(ifr+1)/tstar);
              } else {
                flux *= (1.0-prad->FitIntPlanckFunc(prad->nu_grid(ifr)/tstar));
              }
            }
            // std::cout << "ang: " << ang << "ir: " << ir(k,j,is-i,ang) << std::endl;
            if (prad->mu(0,k,j,is-i,ang) > miux_min) {
              ir(k,j,is-i,ang) = flux / sum_weight;
              // std::cout << "ang: " << ang << "ir: " << ir(k,j,is-i,ang) << std::endl;
            } else if (prad->mu(0,k,j,is-i,ang) < 0.0) {
              ir(k,j,is-i,ang) = 0.0;
            } else {
              ir(k,j,is-i,ang) = ir(k,j,is,ang);
            }
          }
        }
      }
    }
  }
  return;
}

void DiskRadOuterX1(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
              const AthenaArray<Real> &w, FaceField &b,
              AthenaArray<Real> &ir,
              Real time, Real dt,
              int is, int ie, int js, int je, int ks, int ke, int ngh) {
  int nang=prad->nang;
  int nfreq=prad->nfreq;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        for (int ifr=0; ifr<nfreq; ++ifr) {
          for(int n=0; n<nang; ++n){
            int ang = ifr*nang + n;
            Real miux=prad->mu(0,k,j,ie+i,ang);
            if (miux > 0.0)
              ir(k,j,ie+i,ang) = ir(k,j,ie,ang);
            else
              ir(k,j,ie+i,ang) = 0.0;
            // ir(k,j,ie+i,ang) = ir(k,j,ie,ang);
          }
        }
      }
    }
  }
  return;
}

void DiskRadInnerX2(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
              const AthenaArray<Real> &w, FaceField &b,
              AthenaArray<Real> &ir,
              Real time, Real dt,
              int is, int ie, int js, int je, int ks, int ke, int ngh) {
  int nang=prad->nang;
  int nfreq=prad->nfreq;
  std::vector<int> opp(nang*nfreq);

  for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=is; i<=ie; ++i) {
        for (int ifr=0; ifr<nfreq; ++ifr) {
          for(int n=0; n<nang; ++n){
            int ang = ifr*nang + n;

            // for (int nopp = 0; nopp < nang; ++nopp) {
            //   int angopp = ifr*nang + nopp;
            //   if ( std::abs(prad->mu(2,k,js,i,ang) - prad->mu(2,k,js,i,angopp)) < 1e-8
            //     && std::abs(prad->mu(1,k,js,i,ang) + prad->mu(1,k,js,i,angopp)) < 1e-8
            //     && std::abs(prad->mu(0,k,js,i,ang) - prad->mu(0,k,js,i,angopp)) < 1e-8) {
            //     opp[ang] = angopp;
            //     break;
            //   }
            // }

            Real miuy=prad->mu(1,k,js-j,i,ang);
            if (miuy < 0.0) {
              ir(k,js-j,i,ang) = ir(k,js,i,ang);
            } else if (miuy > 0.0){
              ir(k,js-j,i,ang) = 0.0;
            } else{
              ir(k,js-j,i,ang) = ir(k,js,i,ang);
            }

            // for (int k=ks; k<=ke; ++k) {
            //   for (int j=1; j<=ngh; ++j) {
            //     for (int i=is; i<=ie; ++i) {
            //       for (int n=0; n<nang; ++n) {
            //         ir(k,js-j,i,n) = 0.0;
            //       }
            //     }
            //   }
            // }

            // ir(k,js-j,i,ang) = ir(k,js,i,ang);

          }
        }
      }
    }
  }
  return;
}

void DiskRadOuterX2(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
              const AthenaArray<Real> &w, FaceField &b,
              AthenaArray<Real> &ir,
              Real time, Real dt,
              int is, int ie, int js, int je, int ks, int ke, int ngh) {
  int nang=prad->nang;
  int nfreq=prad->nfreq;
  std::vector<int> opp(nang*nfreq);

  for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=is; i<=ie; ++i) {
        for (int ifr=0; ifr<nfreq; ++ifr) {
          for(int n=0; n<nang; ++n){
            int ang = ifr*nang + n;

            // for (int nopp = 0; nopp < nang; ++nopp) {
            //   int angopp = ifr*nang + nopp;
            //   if ( std::abs(prad->mu(2,k,je,i,ang) - prad->mu(2,k,je,i,angopp)) < 1e-8
            //     && std::abs(prad->mu(1,k,je,i,ang) + prad->mu(1,k,je,i,angopp)) < 1e-8
            //     && std::abs(prad->mu(0,k,je,i,ang) - prad->mu(0,k,je,i,angopp)) < 1e-8 ) {
            //     opp[ang] = angopp;
            //     break;
            //   }
            // }

            Real miuy=prad->mu(1,k,je+j,i,ang);
            if (miuy > 0.0){
              ir(k,je+j,i,ang) = ir(k,je,i,ang);
            } else if (miuy < 0.0){
              ir(k,je+j,i,ang) = 0.0;
            } else {
              ir(k,je+j,i,ang) = ir(k,je,i,ang);
            }

            // for (int k=ks; k<=ke; ++k) {
            //   for (int j=1; j<=ngh; ++j) {
            //     for (int i=is; i<=ie; ++i) {
            //       for (int n=0; n<nang; ++n) {
            //         ir(k,je+j,i,n) = 0.0;
            //       }
            //     }
            //   }
            // }

            // ir(k,je+j,i,ang) = ir(k,je,i,ang);

          }
        }
      }
    }
  }
  return;
}

//void DiskRadInnerX3(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
//              const AthenaArray<Real> &w, FaceField &b,
//              AthenaArray<Real> &ir,
//              Real time, Real dt,
//              int is, int ie, int js, int je, int ks, int ke, int ngh) {
//  int nang=prad->nang;
//  int nfreq=prad->nfreq;
//  for (int k=1; k<=ngh; ++k) {
//    for (int j=js; j<=je; ++j) {
//      for (int i=is; i<=ie; ++i) {
//        for (int ifr=0; ifr<nfreq; ++ifr) {
//          for(int n=0; n<nang; ++n){
//            int ang = ifr*nang + n;
//            Real& miux=prad->mu(0,ks-k,j,i,n);
//            if (miux < 0.0)
//              ir(ks-k,j,i,ang) = ir(ks,j,i,n);
//            else
//              ir(ks-k,j,i,ang) = 0.0;
//          }
//        }
//      }
//    }
//  }
//  return;
//}
//
//void DiskRadOuterX3(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
//              const AthenaArray<Real> &w, FaceField &b,
//              AthenaArray<Real> &ir,
//              Real time, Real dt,
//              int is, int ie, int js, int je, int ks, int ke, int ngh) {
//  int nang=prad->nang;
//  int nfreq=prad->nfreq;
//  for (int k=1; k<=ngh; ++k) {
//    for (int j=js; j<=je; ++j) {
//      for (int i=is; i<=ie; ++i) {
//        for (int ifr=0; ifr<nfreq; ++ifr) {
//          for(int n=0; n<nang; ++n){
//            int ang = ifr*nang + n;
//            Real& miux=prad->mu(0,ke+k,j,i,n);
//            if (miux > 0.0)
//              ir(ke+k,j,i,ang) = ir(ke,j,i,n);
//            else
//              ir(ke+k,j,i,ang) = 0.0;
//          }
//        }
//      }
//    }
//  }
//  return;
//}