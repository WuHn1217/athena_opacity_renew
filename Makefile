# Template for Athena++ Makefile
# The 'configure.py' python script uses this template to create the actual Makefile

# Files for conditional compilation

PROBLEM_FILE = inner_disk.cpp
COORDINATES_FILE = spherical_polar.cpp
EOS_FILE = adiabatic_hydro.cpp
GENERAL_EOS_FILE = noop.cpp
RSOLVER_FILE = hllc.cpp
RSOLVER_DIR = hydro/
MPIFFT_FILE =  
CHEMNET_FILE = src/chemistry/network/none.cpp
CHEMISTRY_FILE = src/chemistry/network_wrapper.cpp src/chemistry/utils/*.cpp
CHEM_ODE_SOLVER_FILE = forward_euler.cpp
CHEMRADIATION_FILE = const.cpp

# General compiler specifications

CXX := mpicxx
CPPFLAGS :=  -I/opt/homebrew/opt/hdf5-mpi/include
CXXFLAGS := -O3 -std=c++11
LDFLAGS :=  -L/opt/homebrew/opt/hdf5-mpi/lib
LDLIBS :=  -lhdf5
GCOV_CMD := gcov

# Preliminary definitions

EXE_DIR := bin/
EXECUTABLE := $(EXE_DIR)athena
SRC_FILES := $(wildcard src/*.cpp) \
	     $(wildcard src/bvals/*.cpp) \
	     $(wildcard src/bvals/cc/*.cpp) \
	     $(wildcard src/bvals/cc/fft_grav/*.cpp) \
	     $(wildcard src/bvals/cc/hydro/*.cpp) \
	     $(wildcard src/bvals/cc/mg/*.cpp) \
             $(wildcard src/bvals/cc/nr_radiation/*.cpp) \
	     $(wildcard src/bvals/fc/*.cpp) \
	     $(wildcard src/bvals/orbital/*.cpp) \
	     $(wildcard src/bvals/sixray/*.cpp) \
	     $(wildcard src/bvals/utils/*.cpp) \
	     src/coordinates/coordinates.cpp \
	     src/coordinates/$(COORDINATES_FILE) \
	     src/eos/general/$(GENERAL_EOS_FILE) \
	     src/eos/$(EOS_FILE) \
	     src/eos/eos_high_order.cpp \
	     src/eos/eos_scalars.cpp \
	     $(wildcard $(CHEMISTRY_FILE)) \
	     $(CHEMNET_FILE) \
	     $(wildcard src/chemistry/$(CHEM_ODE_SOLVER_FILE)) \
	     $(wildcard src/chem_rad/*.cpp) \
	     $(wildcard src/chem_rad/integrators/$(CHEMRADIATION_FILE)) \
	     $(wildcard src/fft/*.cpp) \
	     $(wildcard src/field/*.cpp) \
	     $(wildcard src/field/field_diffusion/*.cpp) \
	     $(wildcard src/gravity/*.cpp) \
	     $(wildcard src/hydro/*.cpp) \
	     $(wildcard src/hydro/srcterms/*.cpp) \
	     $(wildcard src/hydro/hydro_diffusion/*.cpp) \
             $(wildcard src/nr_radiation/*.cpp) \
             $(wildcard src/nr_radiation/integrators/*.cpp) \
             $(wildcard src/nr_radiation/integrators/srcterms/*.cpp) \
             $(wildcard src/nr_radiation/implicit/*.cpp) \
             $(wildcard src/cr/*.cpp) \
             $(wildcard src/cr/integrators/*.cpp) \
             $(wildcard src/crdiffusion/*.cpp) \
	     src/hydro/rsolvers/$(RSOLVER_DIR)$(RSOLVER_FILE) \
	     $(wildcard src/inputs/*.cpp) \
	     $(wildcard src/mesh/*.cpp) \
	     $(wildcard src/multigrid/*.cpp) \
	     $(wildcard src/orbital_advection/*.cpp) \
	     $(wildcard src/outputs/*.cpp) \
	     src/pgen/default_pgen.cpp \
	     src/pgen/$(PROBLEM_FILE) \
	     $(wildcard src/reconstruct/*.cpp) \
	     $(wildcard src/scalars/*.cpp) \
	     $(wildcard src/task_list/*.cpp) \
	     $(wildcard src/utils/*.cpp) \
	     $(wildcard src/units/*.cpp) \
	     $(MPIFFT_FILE)
OBJ_DIR := obj/
OBJ_FILES := $(addprefix $(OBJ_DIR),$(notdir $(SRC_FILES:.cpp=.o)))
GCOV_FILES := $(notdir $(addsuffix .gcov,$(SRC_FILES)))
GCDA_FILES := $(wildcard $(OBJ_DIR)/*.gcda)
SRC_PREFIX := src/
SRC_DIRS := $(dir $(SRC_FILES))
VPATH := $(SRC_DIRS)

# Generally useful targets

.PHONY : all dirs clean

all : dirs $(EXECUTABLE)

objs : dirs $(OBJ_FILES)

dirs : $(EXE_DIR) $(OBJ_DIR)

# Placing gcov target in the Makefile in order to easily collect all SRC_FILES w/ correct paths

gcov : dirs $(GCOV_FILES)

# For debugging variables in Makefile, e.g. by "make print-GCOV_FILES"

print-%  : ; @echo $* = $($*)

$(EXE_DIR):
	mkdir -p $(EXE_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Link objects into executable

$(EXECUTABLE) : $(OBJ_FILES)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $(OBJ_FILES) $(LDFLAGS) $(LDLIBS)

# Create objects from source files

$(OBJ_DIR)%.o : %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# Process .gcno and .gcda files from obj/ into .cpp.gcov files (and .hpp.gcov, .h.gcov) in root directory
# Rerun Gcov on all files if a single .gcda changes. Other options to consider: --preserve-paths -abcu
./%.cpp.gcov : %.cpp $(OBJ_DIR)/%.gcno $(GCDA_FILES)
	$(GCOV_CMD)  --relative-only --source-prefix=$(SRC_PREFIX) --object-directory=$(OBJ_DIR) $<

# Cleanup

clean :
	rm -rf $(OBJ_DIR)*
	rm -rf $(EXECUTABLE)
	rm -rf *.gcov
