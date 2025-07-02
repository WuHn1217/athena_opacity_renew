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
#include <numbers>    // pi

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
Real gm0, r0, rho0, tem0, dslope, p0_over_r0, pslope, Omega0, gamma_gas, nu_iso, alpha, dens0, l0, opacity, rstar, tstar;
Real dfloor, tceil;
Real kappa_pm0, kappa_rm0;
int nfreq, nrho, ntem, user_fre;
AthenaArray<Real> rho_table, tem_table, fre_table;
AthenaArray<Real> kappa_pm_table;
AthenaArray<Real> kappa_rm_table;
void SetFrequencies(NRRadiation *prad);
void GetOpacities(const Real rho, const Real tem, const int fre_index, Real &kappa_pm, Real &kappa_rm);
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
void DiskOpacity(MeshBlock *pmb, AthenaArray<Real> &prim);

using CalOpacity = void (*)(const Real, const Real, const int, Real&, Real&);
CalOpacity cal_opacity = nullptr;

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Get parameters for gravitatonal potential of central point mass
  gm0 = pin->GetOrAddReal("problem","gm0", 1.0);
  r0 = pin->GetOrAddReal("problem","r0", 1.0);

  // Get parameters for initial density and velocity
  rho0 = pin->GetReal("problem","rho0");
  dslope = pin->GetOrAddReal("problem","dslope", 0.0);

  // Get parameters of initial pressure and cooling parameters
  if (NON_BAROTROPIC_EOS) {
    p0_over_r0 = pin->GetOrAddReal("problem","p0_over_r0", 0.0);
    pslope = pin->GetOrAddReal("problem","pslope", 0.0);
    gamma_gas = pin->GetReal("hydro","gamma");
  } else {
    p0_over_r0=SQR(pin->GetReal("hydro","iso_sound_speed"));
  }
  Real float_min = std::numeric_limits<float>::min();
  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(float_min)));

  Omega0 = pin->GetOrAddReal("orbital_advection","Omega0", 0.0);

  l0 = pin->GetReal("radiation", "length_unit");
  dens0 = pin->GetReal("radiation", "density_unit");
  tem0 = pin->GetReal("radiation", "T_unit");
  user_fre = pin->GetOrAddInteger("radiation", "user_fre", 0);
  nfreq = pin->GetOrAddReal("radiation", "n_frequency", 1);
  nrho = pin->GetOrAddReal("radiation", "n_density", 1);
  ntem = pin->GetOrAddReal("radiation", "n_temperature", 1);

  alpha = pin -> GetOrAddReal("problem", "alpha", 0.0);
  nu_iso = pin -> GetOrAddReal("problem", "nu_iso", 0.0);
  opacity = pin->GetOrAddReal("problem", "opacity", 1.0);
  rstar = pin->GetOrAddReal("problem", "rstar", 0.0);
  tstar = pin->GetOrAddReal("problem", "tstar", 0.0);
  rstar = rstar / l0;
  tstar = tstar / tem0;

  opacity *= dens0*l0;
  kappa_pm0 = opacity;
  kappa_rm0 = opacity;

  tceil = pin->GetOrAddReal("problem","tceil", 1e5);
  tceil = tceil / tem0;

  if (nu_iso > 0.0) {
    EnrollViscosityCoefficient(Viscosity);
  }

  if (tceil > 0.0) {
    EnrollUserExplicitSourceFunction(Source);
  }

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
  
  std::stringstream msg;
  if (nfreq > 1) {
    int frho_lines = 0;
    int ftem_lines = 0;
    std::string line;
    FILE *frho_table = fopen("../opa_table/sparse/rho_table.txt", "r");
    FILE *ftem_table = fopen("../opa_table/sparse/tem_table.txt", "r");
    FILE *fkappa_pm_table = fopen("../opa_table/sparse/kappa_pm_table.txt", "r");
    FILE *fkappa_rm_table = fopen("../opa_table/sparse/kappa_rm_table.txt", "r");
    kappa_pm_table.NewAthenaArray(nrho+1, ntem+1, nfreq-1);
    kappa_rm_table.NewAthenaArray(nrho+1, ntem+1, nfreq-1);
    rho_table.NewAthenaArray(nrho);
    tem_table.NewAthenaArray(ntem);

    // Open input table files
    if (frho_table == nullptr) {
      msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl
          << "Unable to open rho_table.txt for frequency-dependent opacities";
      ATHENA_ERROR(msg);

      return;
    }
    if (ftem_table == nullptr) {
      msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl
          << "Unable to open tem_table.txt for frequency-dependent opacities";
      ATHENA_ERROR(msg);

      return;
    }
    if (fkappa_pm_table == nullptr) {
      msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl
          << "Unable to open kappa_pm_table.txt for frequency-dependent opacities";
      ATHENA_ERROR(msg);

      return;
    }
    if (fkappa_rm_table == nullptr) {
      msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl
          << "Unable to open kappa_rm_table.txt for frequency-dependent opacities";
      ATHENA_ERROR(msg);

      return;
    }

    // Check table sizes against input parameters
    std::ifstream rho_ifstream("../opa_table/sparse/rho_table.txt");
    while (std::getline(rho_ifstream, line))
      ++frho_lines;
    if (frho_lines != nrho) {
      msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl
          << "rho_table.txt size does not match `n_density` input parameter";
      ATHENA_ERROR(msg);

      return;
    } else {
      rho_ifstream.close();
    }

    std::ifstream tem_ifstream("../opa_table/sparse/tem_table.txt");
    while (std::getline(tem_ifstream, line))
      ++ftem_lines;
    if (ftem_lines != ntem) {
      msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl
          << "tem_table.txt size does not match `n_temperature` input parameter";
      ATHENA_ERROR(msg);

      return;
    } else {
      tem_ifstream.close();
    }

    if (user_fre == 1) {
      std::ifstream fre_ifstream("../opa_table/sparse/fre_table.txt");

      if (!fre_ifstream.is_open()) {
      msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl
          << "Unable to open fre_table.txt for user-defined frequency groups";
      ATHENA_ERROR(msg);

      return;
      } else {
        int ffre_lines = 0;

        while (std::getline(fre_ifstream, line))
          ++ffre_lines;
        if (ffre_lines != nfreq) {
          msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl
              << "fre_table.txt size inconsistent with `n_frequency` input parameter";
          ATHENA_ERROR(msg);

          return;
        } else {
          fre_ifstream.close();
        }
      }
    }


    // Read file values into array tables

    for (int i = 0; i < nrho; ++i) {
      fscanf(frho_table, "%le", &rho_table(i));
    }
    for (int j = 0; j < ntem; ++j) {
      fscanf(ftem_table, "%le", &tem_table(j));
    }
    int ii, jj, kk;
    Real val;
    while (fscanf(fkappa_pm_table, "%d %d %d %le", &ii, &jj, &kk, &val) == 4) {
      kappa_pm_table(ii, jj, kk) = val * (l0 * dens0);
    }
    while (fscanf(fkappa_rm_table, "%d %d %d %le", &ii, &jj, &kk, &val) == 4) {
      kappa_rm_table(ii, jj, kk) = val * (l0 * dens0);
    }
    
    for (int k = 0; k < nfreq-1; ++k) {
      for (int i = 0; i < nrho; ++i) {
        kappa_pm_table(i, ntem, k) = kappa_pm_table(i, ntem-1, k);
        kappa_rm_table(i, ntem, k) = kappa_rm_table(i, ntem-1, k);
      }
    }
    for (int k = 0; k < nfreq-1; ++k) {
      for (int j = 0; j < ntem+1; ++j) {
        kappa_pm_table(nrho, j, k) = kappa_pm_table(nrho-1, j, k);
        kappa_rm_table(nrho, j, k) = kappa_rm_table(nrho-1, j, k);
      }
    }

    // for (int i=0; i<nrho; ++i) {
    //   fscanf(frho_table, "%le", &(rho_table(i)));
    //   rho_table(i) = rho_table(i) / (dens0);
    //   for (int j=0; j<ntem; ++j) {
    //     fscanf(ftem_table, "%le", &(tem_table(j)));
    //     tem_table(j) = tem_table(j) / (tem0);
    //     for (int k=0; k<nfreq-1; ++k) {
    //       Real val;
    //       while (fscanf(fkappa_pm_table, "%d %d %d %le", &i, &j, &k, &val) == 4) {
    //         kappa_pm_table(i, j, k) = val;
    //         kappa_pm_table(i, j, k) = kappa_pm_table(i, j, k) * (l0*dens0);
    //       }
    //       while (fscanf(fkappa_rm_table, "%d %d %d %le", &i, &j, &k, &val) == 4) {
    //         kappa_rm_table(i, j, k) = val;
    //         kappa_rm_table(i, j, k) = kappa_rm_table(i, j, k) * (l0*dens0);
    //       }
    //     }
    //   }
    // }

    fclose(fkappa_pm_table);
    fclose(fkappa_rm_table);
    fclose(frho_table);
    fclose(ftem_table);

    if (user_fre == 1) {
      Real k_b = 1.3807e-16;  // Boltzmann constant [erg/K]
      Real h = 6.626196e-27;  // Planck constant [erg s]
      FILE *ffre_table = fopen("../opa_table/sparse/fre_table.txt", "r");
      fre_table.NewAthenaArray(nfreq);
      for (int i=0; i<nfreq-1; ++i) {
        fscanf(ffre_table, "%le", &(fre_table(i)));
        fre_table(i) /= k_b*tem0/h;
      }
      fclose(ffre_table);
    }
  }

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  // enroll user-defined opacity function
  if (NR_RADIATION_ENABLED || IM_RADIATION_ENABLED) {
    pnrrad->EnrollOpacityFunction(DiskOpacity);
    if (user_fre == 1)
      pnrrad->EnrollFrequencyFunction(SetFrequencies);
  }

  AllocateUserOutputVariables(1);
  AllocateRealUserMeshBlockDataField(1);
  ruser_meshblock_data[0].NewAthenaArray(block_size.nx3, block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST);

  cal_opacity = GetOpacities;

  return;
}

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

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin)
{
  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      for(int i=is; i<=ie; i++) {
        user_out_var(0,k,j,i) = ruser_meshblock_data[0](k,j,i);
      }
    }
  }
}

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
  int il = pmb->is; int jl = pmb->js; int kl = pmb->ks;
  int iu = pmb->ie; int ju = pmb->je; int ku = pmb->ke;

  //Include ghost zones
  il -= NGHOST;
  iu += NGHOST;
  if(ju > jl){
  jl -= NGHOST;
  ju += NGHOST;
  }
  if(ku > kl){
  kl -= NGHOST;
  ku += NGHOST;
  }

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        const Real gas_pre = prim(IPR, k, j, i);
        const Real gas_rho = prim(IDN, k, j, i);
        Real gamma          = pmb->peos->GetGamma();
        Real gas_tem        = gas_pre / gas_rho;
        if (tceil > 0) {
          if (gas_tem > tceil) {
            cons(IEN,k,j,i) -= prim(IDN,k,j,i) * (gas_tem - tceil) / (gamma - 1.0);
          }
        } else {
          return;
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

void SetFrequencies(NRRadiation *prad) {
  prad->nu_grid(0) = 0.0;

  for(int i=0; i<nfreq; ++i) {
    prad->nu_grid(i+1) = fre_table(i);
    prad->nu_cen(i) = (prad->nu_grid(i) + prad->nu_grid(i+1))/2;
    prad->delta_nu(i) = prad->nu_grid(i+1) - prad->nu_grid(i);
  }
}

void GetOpacities(const Real rho, const Real tem, const int fre_index,
                  Real &kappa_pm, Real &kappa_rm) {
  int rho_index, tem_index;
  Real r, t;
  
  if (rho < 1e-14) {
    rho_index = 0;
  } else if (rho > 1.0) {
    rho_index = 299;
  } else {
    rho_index = std::floor((299.0 / 14.0) * log10(rho) + 299);
  }

  if (tem < 1.0) {
    tem_index = 0;
  } else if (tem > 1e7) {
    tem_index = 299;
  } else {
    tem_index = std::floor((299.0 / 7.0) * log10(tem));
  }

  // kappa_pm = kappa_pm_table(rho_index, tem_index, fre_index);
  // kappa_rm = kappa_rm_table(rho_index, tem_index, fre_index);

  Real log_rho = std::log10(rho);
  Real log_tem = std::log10(tem);

  Real k00_pm = std::log10(kappa_pm_table(rho_index, tem_index, fre_index));
  Real k01_pm = std::log10(kappa_pm_table(rho_index, tem_index+1, fre_index));
  Real k10_pm = std::log10(kappa_pm_table(rho_index+1, tem_index, fre_index));
  Real k11_pm = std::log10(kappa_pm_table(rho_index+1, tem_index+1, fre_index));

  Real k00_rm = std::log10(kappa_rm_table(rho_index, tem_index, fre_index));
  Real k01_rm = std::log10(kappa_rm_table(rho_index, tem_index+1, fre_index));
  Real k10_rm = std::log10(kappa_rm_table(rho_index+1, tem_index, fre_index));
  Real k11_rm = std::log10(kappa_rm_table(rho_index+1, tem_index+1, fre_index));

  // Real k00_pm = kappa_pm_table(rho_index, tem_index, fre_index);
  // Real k01_pm = kappa_pm_table(rho_index, tem_index+1, fre_index);
  // Real k10_pm = kappa_pm_table(rho_index+1, tem_index, fre_index);
  // Real k11_pm = kappa_pm_table(rho_index+1, tem_index+1, fre_index);

  // Real k00_rm = kappa_rm_table(rho_index, tem_index, fre_index);
  // Real k01_rm = kappa_rm_table(rho_index, tem_index+1, fre_index);
  // Real k10_rm = kappa_rm_table(rho_index+1, tem_index, fre_index);
  // Real k11_rm = kappa_rm_table(rho_index+1, tem_index+1, fre_index);

  if (rho < 1e-14) {
    r = 0.0;
  } else if (rho > 1.0) {
    r = 1.0;
  } else {
    r = (log_rho - std::log10(rho_table(rho_index))) / (std::log10(rho_table(rho_index+1)) - std::log10(rho_table(rho_index)));
  }

  if (tem < 1.0) {
    t = 0.0;
  } else if (tem > 1e7) {
    t = 1.0;
  } else {
    t = (log_tem - std::log10(tem_table(tem_index))) / (std::log10(tem_table(tem_index+1)) - std::log10(tem_table(tem_index)));
  }

  kappa_pm = (1-t) * (1-r) * k00_pm + t * (1-r) * k01_pm + (1-t) * r * k10_pm + t * r * k11_pm;
  kappa_rm = (1-t) * (1-r) * k00_rm + t * (1-r) * k01_rm + (1-t) * r * k10_rm + t * r * k11_rm;
  kappa_pm = std::pow(10.0, kappa_pm);
  kappa_rm = std::pow(10.0, kappa_rm);

  return;
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
}

void DiskRadInnerX1(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
              const AthenaArray<Real> &w, FaceField &b,
              AthenaArray<Real> &ir,
              Real time, Real dt,
              int is, int ie, int js, int je, int ks, int ke, int ngh) {
  Real sum_weight;      // sum of weights and miu_x throughout all angles
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
            if (nfreq == 1) {
              flux *= 1.0;
            } else {
              if (ifr < nfreq-1) {
                flux *= prad->IntPlanckFunc(prad->nu_grid(ifr)/tstar,
                                        prad->nu_grid(ifr+1)/tstar);
              } else {
                flux *= (1.0-prad->FitIntPlanckFunc(prad->nu_grid(ifr)/tstar));
              }
            }
            if (prad->mu(0,k,j,is-i,ang) > miux_min) {
              ir(k,j,is-i,ang) = flux / sum_weight;
            } else if (prad->mu(0,k,j,is-i,ang) < 0.0) {
              ir(k,j,is-i,ang) = ir(k,j,is,ang);
            } else {
              ir(k,j,is-i,ang) = 0.0;
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
            Real& miux=prad->mu(0,k,j,ie+i,ang);
            if (miux > 0.0)
              ir(k,j,ie+i,ang) = ir(k,j,ie,ang);
            else
              ir(k,j,ie+i,ang) = 0.0;
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
  for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=is; i<=ie; ++i) {
        for (int ifr=0; ifr<nfreq; ++ifr) {
          for(int n=0; n<nang; ++n){
            int ang = ifr*nang + n;
            Real miuy=prad->mu(1,k,js-j,i,ang);
            if (miuy < 0.0) {
              ir(k,js-j,i,ang) = ir(k,js,i,ang);
            } else if (miuy > 0.0){
              ir(k,js-j,i,ang) = 0.0;
            } else{
              ir(k,js-j,i,ang) = ir(k,js,i,ang);
            }
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
  for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=is; i<=ie; ++i) {
        for (int ifr=0; ifr<nfreq; ++ifr) {
          for(int n=0; n<nang; ++n){
            int ang = ifr*nang + n;
            Real miuy=prad->mu(1,k,je+j,i,ang);
            if (miuy > 0.0){
              ir(k,je+j,i,ang) = ir(k,je,i,ang);
            } else if (miuy < 0.0){
              ir(k,je+j,i,ang) = 0.0;
            } else {
              ir(k,je+j,i,ang) = ir(k,je,i,ang);
            }
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

void DiskOpacity(MeshBlock *pmb, AthenaArray<Real> &prim) {
  NRRadiation *prad = pmb->pnrrad;
  int il = pmb->is; int jl = pmb->js; int kl = pmb->ks;
  int iu = pmb->ie; int ju = pmb->je; int ku = pmb->ke;
  int fre_index;              // frequency index
  Real tem_gas, rho_gas;      // gas temperature and density
  Real kappa_pm = kappa_pm0;    
  Real kappa_rm = kappa_rm0;
  
  //Include ghost zones in upper/lower directional limits
  il -= NGHOST;
  iu += NGHOST;
  if(ju > jl){
    jl -= NGHOST;
    ju += NGHOST;
  }
  if(ku > kl){
    kl -= NGHOST;
    ku += NGHOST;
  }

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        for (int ifr=0; ifr<nfreq; ++ifr) {
          if (nfreq > 1) {
            rho_gas = prim(IDN,k,j,i)*dens0;
            tem_gas = (prim(IPR,k,j,i)/prim(IDN,k,j,i))*tem0;
            if (ifr == 0) {
              fre_index = ifr;
            } else {
              fre_index = ifr - 1;
            }
            GetOpacities(rho_gas, tem_gas, fre_index, kappa_pm, kappa_rm);
          } else {
            std::stringstream msg;
            msg << "### FATAL ERROR in function [DiskOpacity]" << std::endl
            << "ERROR: nfreq too small for fre_index logic!";
            ATHENA_ERROR(msg);
          }

          prad->sigma_s(k,j,i,ifr) = 0;
          prad->sigma_a(k,j,i,ifr) = prim(IDN,k,j,i)*kappa_rm;
          prad->sigma_pe(k,j,i,ifr) = prim(IDN,k,j,i)*kappa_pm;
          if (prad->set_source_flag == 0) {                   
            prad->sigma_p(k,j,i,ifr) = 0;
          } else {
            prad->sigma_p(k,j,i,ifr) = prim(IDN,k,j,i)*kappa_pm;
          }
        }
      }
    }
  }
}