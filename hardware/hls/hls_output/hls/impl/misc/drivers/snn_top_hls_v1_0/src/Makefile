# ==============================================================
# Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
# Tool Version Limit: 2025.11
# Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
# Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
# 
# ==============================================================
COMPILER=
ARCHIVER=
CP=cp
RM=rm
COMPILER_FLAGS=
EXTRA_COMPILER_FLAGS=
ARCHIVER_FLAGS=

CORENAME=snn_top_hls

RELEASEDIR=../../../lib
OBJDIR=$(RELEASEDIR)
LIBDIR=$(RELEASEDIR)
INCLUDEDIR=../../../include
INCLUDES=-I./. -I$(INCLUDEDIR)

# Source files
SRCS := $(wildcard *.c)
HEADERS := $(wildcard *.h)

# Generated files
LIB := $(LIBDIR)/libxil.a
OBJS := $(addprefix $(OBJDIR)/,$(addsuffix .o,$(basename $(SRCS))))
RELEASE_HEADERS := $(addprefix $(INCLUDEDIR)/,$(notdir $(HEADERS)))

# Rules to compile and link
$(OBJDIR)/%.o: %.c
	$(COMPILER) $(COMPILER_FLAGS) $(EXTRA_COMPILER_FLAGS) $(INCLUDES) -o $@ $<
$(LIB): $(OBJS)
	$(ARCHIVER) $(ARCHIVER_FLAGS) -r $@ $^

# Rule to release include files
$(INCLUDEDIR)/%.h: %.h
	$(CP) $< $@

# Dependencies between sources and objects
$(OBJS): $(SRCS) | $(HEADERS)

# Named targets for use by others
include: $(RELEASE_HEADERS)
includes: include
libs: $(LIB) banner
objs: $(OBJS)

# Helper targets
clean:
	$(RM) -f $(LIB) $(OBJS) $(RELEASE_HEADERS) 
banner:
	echo "Building $(CORENAME)"

.PHONY: clean include objs libs includes banner


