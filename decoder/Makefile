SIM ?= verilator
TOPLEVEL_LANG ?= verilog

VERILOG_SOURCES += $(PWD)/ibex_pkg.sv \
	$(PWD)/ibex_decoder_wrap.sv \
	$(PWD)/ibex_decoder.sv

TOPLEVEL = ibex_decoder_wrap
MODULE = test_decoder

EXTRA_ARGS += --trace-fst --trace-structs \
	-Wno-fatal -CFLAGS "-std=c++14 -Wall -O3"

include $(shell cocotb-config --makefiles)/Makefile.sim
