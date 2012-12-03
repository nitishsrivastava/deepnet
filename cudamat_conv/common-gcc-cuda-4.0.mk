################################################################################
#
# Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property and 
# proprietary rights in and to this software and related documentation. 
# Any use, reproduction, disclosure, or distribution of this software 
# and related documentation without an express license agreement from
# NVIDIA Corporation is strictly prohibited.
#
# Please refer to the applicable NVIDIA end user license agreement (EULA) 
# associated with this source code for terms and conditions that govern 
# your use of this NVIDIA software.
#
################################################################################
#
# Common build script for CUDA source projects for Linux and Mac platforms
#
################################################################################

.SUFFIXES : .cu .cu_dbg.o .c_dbg.o .cpp_dbg.o .cu_rel.o .c_rel.o .cpp_rel.o .cubin .ptx

# Add new SM Versions here as devices with new Compute Capability are released
SM_VERSIONS   := 10 11 12 13 20 21

CUDA_INSTALL_PATH ?= /usr/local/cuda

ifdef cuda-install
	CUDA_INSTALL_PATH := $(cuda-install)
endif

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# 'linux' is output for Linux system, 'darwin' for OS X
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))
ifneq ($(DARWIN),)
   SNOWLEOPARD = $(strip $(findstring 10.6, $(shell egrep "<string>10\.6" /System/Library/CoreServices/SystemVersion.plist)))
endif

# detect 32-bit or 64-bit platform
HP_64 = $(shell uname -m | grep 64)
OSARCH= $(shell uname -m)

# Basic directory setup for SDK
# (override directories only if they are not already defined)
SRCDIR     ?= 
ROOTDIR    ?= $(CUDA_SDK_PATH)
ROOTBINDIR ?= bin
BINDIR     ?= $(ROOTBINDIR)/$(OSLOWER)
ROOTOBJDIR ?= obj
LIBDIR     := $(ROOTDIR)/C/lib
COMMONDIR  := $(ROOTDIR)/C/common
SHAREDDIR  := $(ROOTDIR)/shared/

# Compilers
NVCC       := $(CUDA_INSTALL_PATH)/bin/nvcc 
CXX        := /usr/bin/g++-4.4 -fPIC
CC         := /usr/bin/gcc-4.4 -fPIC
LINK       := /usr/bin/g++-4.4 -fPIC

# Includes
INCLUDES  += -I. -I$(CUDA_INSTALL_PATH)/include -I$(COMMONDIR)/inc -I$(SHAREDDIR)/inc

# Warning flags
CXXWARN_FLAGS := \
	-W -Wall \
	-Wimplicit \
	-Wswitch \
	-Wformat \
	-Wchar-subscripts \
	-Wparentheses \
	-Wmultichar \
	-Wtrigraphs \
	-Wpointer-arith \
	-Wcast-align \
	-Wreturn-type \
	-Wno-unused-function \
	$(SPACE)

CWARN_FLAGS := $(CXXWARN_FLAGS) \
	-Wstrict-prototypes \
	-Wmissing-prototypes \
	-Wmissing-declarations \
	-Wnested-externs \
	-Wmain \

# architecture flag for nvcc and gcc compilers build
CUBIN_ARCH_FLAG :=
CXX_ARCH_FLAGS  :=
NVCCFLAGS       :=
LIB_ARCH        := $(OSARCH)

# Determining the necessary Cross-Compilation Flags
# 32-bit OS, but we target 64-bit cross compilation
ifeq ($(x86_64),1) 
    NVCCFLAGS       += -m64
    LIB_ARCH         = x86_64
    CUDPPLIB_SUFFIX  = x86_64
    ifneq ($(DARWIN),)
         CXX_ARCH_FLAGS += -arch x86_64
    else
         CXX_ARCH_FLAGS += -m64
    endif
else 
# 64-bit OS, and we target 32-bit cross compilation
    ifeq ($(i386),1)
        NVCCFLAGS       += -m32
        LIB_ARCH         = i386
        CUDPPLIB_SUFFIX  = i386
        ifneq ($(DARWIN),)
             CXX_ARCH_FLAGS += -arch i386
        else
             CXX_ARCH_FLAGS += -m32
        endif
    else 
        ifeq "$(strip $(HP_64))" ""
            LIB_ARCH        = i386
            CUDPPLIB_SUFFIX = i386
            NVCCFLAGS      += -m32
            ifneq ($(DARWIN),)
               CXX_ARCH_FLAGS += -arch i386
            else
               CXX_ARCH_FLAGS += -m32
            endif
        else
            LIB_ARCH        = x86_64
            CUDPPLIB_SUFFIX = x86_64
            NVCCFLAGS      += -m64
            ifneq ($(DARWIN),)
               CXX_ARCH_FLAGS += -arch x86_64
            else
               CXX_ARCH_FLAGS += -m64
            endif
        endif
    endif
endif

# Compiler-specific flags (by default, we always use sm_10 and sm_20), unless we use the SMVERSION template
#GENCODE_SM10 := -gencode=arch=compute_10,code=\"sm_10,compute_10\"
#GENCODE_SM20 := -gencode=arch=compute_20,code=\"sm_20,compute_20\"

CXXFLAGS  += $(CXXWARN_FLAGS) $(CXX_ARCH_FLAGS)
CFLAGS    += $(CWARN_FLAGS) $(CXX_ARCH_FLAGS)
LINKFLAGS += -Wl,-no-undefined
LINK      += $(LINKFLAGS) $(CXX_ARCH_FLAGS)

# This option for Mac allows CUDA applications to work without requiring to set DYLD_LIBRARY_PATH
ifneq ($(DARWIN),)
   LINK += -Xlinker -rpath $(CUDA_INSTALL_PATH)/lib
endif

# Common flags
COMMONFLAGS += $(INCLUDES) -DUNIX

# Debug/release configuration
ifeq ($(dbg),1)
	COMMONFLAGS += -g
	NVCCFLAGS   += -D_DEBUG -G
	CXXFLAGS    += -D_DEBUG
	CFLAGS      += -D_DEBUG
	BINSUBDIR   := debug
	LIBSUFFIX   := D
else 
	COMMONFLAGS += -O2 
	BINSUBDIR   := release
	LIBSUFFIX   := 
	NVCCFLAGS   += --compiler-options -fno-strict-aliasing
	CXXFLAGS    += -fno-strict-aliasing
	CFLAGS      += -fno-strict-aliasing
endif

ifneq ($(exec),1)
    NVCCFLAGS += --compiler-options '-fPIC'
	LINK += -shared
endif

# Add profiler output
ifeq ($(prof),1)
	NVCCFLAGS += --ptxas-options=-v
endif

# architecture flag for cubin build
CUBIN_ARCH_FLAG :=

# OpenGL is used or not (if it is used, then it is necessary to include GLEW)
ifeq ($(USEGLLIB),1)
    ifneq ($(DARWIN),)
        OPENGLLIB := -L/System/Library/Frameworks/OpenGL.framework/Libraries 
        OPENGLLIB += -lGL -lGLU $(COMMONDIR)/lib/$(OSLOWER)/libGLEW.a
    else
# this case for linux platforms
	OPENGLLIB := -lGL -lGLU -lX11 -lXi -lXmu
# check if x86_64 flag has been set, otherwise, check HP_64 is i386/x86_64
        ifeq ($(x86_64),1) 
	       OPENGLLIB += -lGLEW_x86_64 -L/usr/X11R6/lib64
        else
             ifeq ($(i386),)
                 ifeq "$(strip $(HP_64))" ""
	             OPENGLLIB += -lGLEW -L/usr/X11R6/lib
                 else
	             OPENGLLIB += -lGLEW_x86_64 -L/usr/X11R6/lib64
                 endif
             endif
        endif
# check if i386 flag has been set, otehrwise check HP_64 is i386/x86_64
        ifeq ($(i386),1)
	       OPENGLLIB += -lGLEW -L/usr/X11R6/lib
        else
             ifeq ($(x86_64),)
                 ifeq "$(strip $(HP_64))" ""
	             OPENGLLIB += -lGLEW -L/usr/X11R6/lib
                 else
	             OPENGLLIB += -lGLEW_x86_64 -L/usr/X11R6/lib64
                 endif
             endif
        endif
    endif
endif

ifeq ($(USEGLUT),1)
    ifneq ($(DARWIN),)
	OPENGLLIB += -framework GLUT
    else
        ifeq ($(x86_64),1)
	     OPENGLLIB += -lglut -L/usr/lib64 
        endif
        ifeq ($(i386),1)
	     OPENGLLIB += -lglut -L/usr/lib 
        endif

        ifeq ($(x86_64),)
            ifeq ($(i386),)  
	        OPENGLLIB += -lglut
            endif
        endif
    endif
endif

ifeq ($(USEPARAMGL),1)
	PARAMGLLIB := -lparamgl_$(LIB_ARCH)$(LIBSUFFIX)
endif

ifeq ($(USERENDERCHECKGL),1)
	RENDERCHECKGLLIB := -lrendercheckgl_$(LIB_ARCH)$(LIBSUFFIX)
endif

ifeq ($(USECUDPP), 1)
    CUDPPLIB := -lcudpp_$(CUDPPLIB_SUFFIX)

    ifeq ($(emu), 1)
        CUDPPLIB := $(CUDPPLIB)_emu
    endif
endif

ifeq ($(USENVCUVID), 1)
     ifneq ($(DARWIN),)
         NVCUVIDLIB := -L../../common/lib/darwin -lnvcuvid
     endif
endif

# Libs
ifneq ($(DARWIN),)
    LIB       += -L$(CUDA_INSTALL_PATH)/lib -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -L$(SHAREDDIR)/lib $(NVCUVIDLIB) 
else
  ifeq "$(strip $(HP_64))" ""
    ifeq ($(x86_64),1)
       LIB       += -L$(CUDA_INSTALL_PATH)/lib64 -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -L$(SHAREDDIR)/lib 
    else
       LIB       += -L$(CUDA_INSTALL_PATH)/lib -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -L$(SHAREDDIR)/lib
    endif
  else
    ifeq ($(i386),1)
       LIB       += -L$(CUDA_INSTALL_PATH)/lib -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -L$(SHAREDDIR)/lib
    else
       LIB       += -L$(CUDA_INSTALL_PATH)/lib64 -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -L$(SHAREDDIR)/lib
    endif
  endif
endif

# If dynamically linking to CUDA and CUDART, we exclude the libraries from the LIB
ifeq ($(USECUDADYNLIB),1)
     LIB := ${OPENGLLIB} $(PARAMGLLIB) $(RENDERCHECKGLLIB) $(CUDPPLIB) ${LIB} -ldl -rdynamic 
else
# static linking, we will statically link against CUDA and CUDART
  ifeq ($(USEDRVAPI),1)
     LIB := -lcuda   ${OPENGLLIB} $(PARAMGLLIB) $(RENDERCHECKGLLIB) $(CUDPPLIB) ${LIB} 
  else
     ifeq ($(emu),1) 
         LIB += -lcudartemu
     else 
         LIB += -lcudart
     endif
     LIB := ${OPENGLLIB} $(PARAMGLLIB) $(RENDERCHECKGLLIB) $(CUDPPLIB) ${LIB}
  endif
endif

ifeq ($(USECUFFT),1)
  ifeq ($(emu),1)
    LIB += -lcufftemu
  else
    LIB += -lcufft
  endif
endif

ifeq ($(USECUBLAS),1)
  ifeq ($(emu),1)
    LIB += -lcublasemu
  else
    LIB += -lcublas
  endif
endif

ifeq ($(USECURAND),1)
    LIB += -lcurand
endif

ifeq ($(USECUSPARSE),1)
  LIB += -lcusparse
endif

# Lib/exe configuration
# Lib/exe configuration
# Lib/exe configuration
ifneq ($(STATIC_LIB),)
	TARGETDIR := $(LIBDIR)
	TARGET   := $(subst .a,_$(LIB_ARCH)$(LIBSUFFIX).a,$(LIBDIR)/$(STATIC_LIB))
	LINKLINE  = ar rucv $(TARGET) $(OBJS)
else
	ifneq ($(OMIT_CUTIL_LIB),1)
		LIB += -lcutil_$(LIB_ARCH)$(LIBSUFFIX) -lshrutil_$(LIB_ARCH)$(LIBSUFFIX)
	endif
	# Device emulation configuration
	ifeq ($(emu), 1)
		NVCCFLAGS   += -deviceemu
		CUDACCFLAGS += 
		BINSUBDIR   := emu$(BINSUBDIR)
		# consistency, makes developing easier
		CXXFLAGS		+= -D__DEVICE_EMULATION__
		CFLAGS			+= -D__DEVICE_EMULATION__
	endif
	TARGETDIR := $(BINDIR)/$(BINSUBDIR)
	TARGET    := $(TARGETDIR)/$(EXECUTABLE)
	LINKLINE  = $(LINK) -o $(TARGET) $(OBJS) $(LIB)
endif

# check if verbose 
ifeq ($(verbose), 1)
	VERBOSE :=
else
	VERBOSE := @
endif

################################################################################
# Check for input flags and set compiler flags appropriately
################################################################################
ifeq ($(fastmath), 1)
	NVCCFLAGS += -use_fast_math
endif

ifeq ($(keep), 1)
	NVCCFLAGS += -keep
	NVCC_KEEP_CLEAN := *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx
endif

ifdef maxregisters
	NVCCFLAGS += -maxrregcount $(maxregisters)
endif

ifeq ($(ptxas), 1)
        NVCCFLAGS += --ptxas-options=-v
endif

# Add cudacc flags
NVCCFLAGS += $(CUDACCFLAGS)

# Add common flags
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS  += $(COMMONFLAGS)
CFLAGS    += $(COMMONFLAGS)

ifeq ($(nvcc_warn_verbose),1)
	NVCCFLAGS += $(addprefix --compiler-options ,$(CXXWARN_FLAGS)) 
	NVCCFLAGS += --compiler-options -fno-strict-aliasing
endif

################################################################################
# Set up object files
################################################################################
OBJDIR := $(ROOTOBJDIR)/$(LIB_ARCH)/$(BINSUBDIR)
OBJS +=  $(patsubst %.cpp,$(OBJDIR)/%.cpp.o,$(CCFILES))
OBJS +=  $(patsubst %.c,$(OBJDIR)/%.c.o,$(CFILES))
OBJS +=  $(patsubst %.cu,$(OBJDIR)/%.cu.o,$(CUFILES))

################################################################################
# Set up cubin output files
################################################################################
CUBINDIR := $(SRCDIR)data
CUBINS +=  $(patsubst %.cu,$(CUBINDIR)/%.cubin,$(notdir $(CUBINFILES)))

################################################################################
# Set up PTX output files
################################################################################
PTXDIR := $(SRCDIR)data
PTXBINS +=  $(patsubst %.cu,$(PTXDIR)/%.ptx,$(notdir $(PTXFILES)))

################################################################################
# Rules
################################################################################
$(OBJDIR)/%.c.o : $(SRCDIR)%.c $(C_DEPS)
	$(VERBOSE)$(CC) $(CFLAGS) -o $@ -c $<

$(OBJDIR)/%.cpp.o : $(SRCDIR)%.cpp $(C_DEPS)
	$(VERBOSE)$(CXX) $(CXXFLAGS) -o $@ -c $<

# Default arch includes gencode for sm_10, sm_20, and other archs from GENCODE_ARCH declared in the makefile
$(OBJDIR)/%.cu.o : $(SRCDIR)%.cu $(CU_DEPS)
	$(VERBOSE)$(NVCC) $(GENCODE_SM10) $(GENCODE_ARCH) $(GENCODE_SM20) $(NVCCFLAGS) $(SMVERSIONFLAGS) -o $@ -c $<

# Default arch includes gencode for sm_10, sm_20, and other archs from GENCODE_ARCH declared in the makefile
$(CUBINDIR)/%.cubin : $(SRCDIR)%.cu cubindirectory
	$(VERBOSE)$(NVCC) $(GENCODE_SM10) $(GENCODE_ARCH) $(GENCODE_SM20) $(CUBIN_ARCH_FLAG) $(NVCCFLAGS) $(SMVERSIONFLAGS) -o $@ -cubin $<

$(PTXDIR)/%.ptx : $(SRCDIR)%.cu ptxdirectory
	$(VERBOSE)$(NVCC) $(CUBIN_ARCH_FLAG) $(NVCCFLAGS) $(SMVERSIONFLAGS) -o $@ -ptx $<

#
# The following definition is a template that gets instantiated for each SM
# version (sm_10, sm_13, etc.) stored in SMVERSIONS.  It does 2 things:
# 1. It adds to OBJS a .cu_sm_XX.o for each .cu file it finds in CUFILES_sm_XX.
# 2. It generates a rule for building .cu_sm_XX.o files from the corresponding 
#    .cu file.
#
# The intended use for this is to allow Makefiles that use common.mk to compile
# files to different Compute Capability targets (aka SM arch version).  To do
# so, in the Makefile, list files for each SM arch separately, like so:
# This will be used over the default rule abov
#
# CUFILES_sm_10 := mycudakernel_sm10.cu app.cu
# CUFILES_sm_12 := anothercudakernel_sm12.cu
#
define SMVERSION_template
#OBJS += $(patsubst %.cu,$(OBJDIR)/%.cu_$(1).o,$(notdir $(CUFILES_$(1))))
OBJS += $(patsubst %.cu,$(OBJDIR)/%.cu_$(1).o,$(notdir $(CUFILES_sm_$(1))))
$(OBJDIR)/%.cu_$(1).o : $(SRCDIR)%.cu $(CU_DEPS)
#	$(VERBOSE)$(NVCC) -o $$@ -c $$< $(NVCCFLAGS)  $(1)
	$(VERBOSE)$(NVCC) -gencode=arch=compute_$(1),code=\"sm_$(1),compute_$(1)\" $(GENCODE_SM20) -o $$@ -c $$< $(NVCCFLAGS)
endef

# This line invokes the above template for each arch version stored in
# SM_VERSIONS.  The call funtion invokes the template, and the eval
# function interprets it as make commands.
$(foreach smver,$(SM_VERSIONS),$(eval $(call SMVERSION_template,$(smver))))

$(TARGET): makedirectories $(OBJS) $(CUBINS) $(PTXBINS) link Makefile
	$(VERBOSE)$(LINKLINE)

link: 
	$(VERBOSE)ln -sf ./bin/linux/release/$(EXECUTABLE) .

cubindirectory:
	$(VERBOSE)mkdir -p $(CUBINDIR)

ptxdirectory:
	$(VERBOSE)mkdir -p $(PTXDIR)

tidy :
	$(VERBOSE)find . | egrep "#" | xargs rm -f
	$(VERBOSE)find . | egrep "\~" | xargs rm -f

clean : tidy
	$(VERBOSE)rm -f *.stub.c *.gpu *.cu.cpp *.i *.ii
	$(VERBOSE)rm -f *.cubin *.ptx *.fatbin.c *.hash
	$(VERBOSE)rm -f *.cudafe1.c *.cudafe2.c *.cudafe1.cpp *.cudafe2.cpp
	$(VERBOSE)rm -f $(OBJS) 
	$(VERBOSE)rm -f $(CUBINS)
	$(VERBOSE)rm -f $(PTXBINS)
	$(VERBOSE)rm -f $(TARGET)
	$(VERBOSE)rm -f $(NVCC_KEEP_CLEAN)
	$(VERBOSE)rm -f $(ROOTBINDIR)/$(OSLOWER)/$(BINSUBDIR)/*.ppm
	$(VERBOSE)rm -f $(ROOTBINDIR)/$(OSLOWER)/$(BINSUBDIR)/*.pgm
	$(VERBOSE)rm -f $(ROOTBINDIR)/$(OSLOWER)/$(BINSUBDIR)/*.bin
	$(VERBOSE)rm -f $(ROOTBINDIR)/$(OSLOWER)/$(BINSUBDIR)/*.bmp

clobber : clean
	$(VERBOSE)rm -rf $(ROOTOBJDIR)
