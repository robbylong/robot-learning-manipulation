# a makefile to generate urdf files from xacro files

# this makefile searches in this same folder for files ending in urdf.xacro and
# runs xacro on each one that it finds

# ----- user defined options ----- #

# define directory structure
SDFDIR = sdf
SEMANTICDIR = semantic
XACRODIR = ../xacro
CONFIGDEP = ../config/gripper.yaml

MAKEFLAGS += -j8 # run on up to 8 parallel cpu cores

# ----- automatically generated variables ----- #

# find urdf.xacro files and create our target file names
XACROS := $(wildcard *.urdf.xacro)
URDFS := $(patsubst %.urdf.xacro, %.urdf, $(XACROS))
SDFS := $(patsubst %.urdf.xacro, $(SDFDIR)/%.sdf, $(XACROS))

# find semantic urdf files
SEMANCTICXACROS := $(wildcard $(SEMANTICDIR)/*.semantic.xacro)
SEMANTICS := $(patsubst $(SEMANTICDIR)/%.semantic.xacro, \
								$(SEMANTICDIR)/%.semantic.urdf, $(SEMANCTICXACROS))

# find xacro files that we depend on for updating changes
DEPS := $(wildcard $(XACRODIR)/*.xacro) $(CONFIGDEP)
SEMANTICDEPS := $(wildcard $(XACRODIR)/$(SEMANTICDIR)/*.xacro) $(CONFIGDEP)

# ----- start of make ----- #

all: $(URDFS) $(SDFS) $(SEMANTICS)

# convert each xacro to urdf using xacro3 script
$(URDFS): %.urdf : %.urdf.xacro $(DEPS)
	./xacro3 $< > $@

# convert each urdf to sdf using the gazebo 'gz sdf' system command
$(SDFS): $(SDFDIR)/%.sdf : %.urdf
	gz sdf -p $< > $@

# convert semantic xacros
$(SEMANTICS): $(SEMANTICDIR)/%.semantic.urdf : $(SEMANTICDIR)/%.semantic.xacro $(SEMANTICDEPS)
	./xacro3 $< > $@

clean:
	rm -f $(URDFS)
	rm -f $(SDFS)
	rm -f $(SEMANTICS)