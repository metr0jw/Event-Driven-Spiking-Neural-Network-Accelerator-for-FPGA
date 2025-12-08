#include <iostream>
#include "hls_stream.h"
#include "hls_directio.h"

using namespace std;

struct __cosim_T_1__ {char data[1];};
extern "C" void fpga_fifo_push_1(__cosim_T_1__* val, hls::stream<__cosim_T_1__>* fifo) {
  fifo->write(*val);
}
extern "C" void fpga_fifo_pop_1(__cosim_T_1__* val, hls::stream<__cosim_T_1__>* fifo) {
  *val = fifo->read();
}
extern "C" bool fpga_fifo_not_empty_1(hls::stream<__cosim_T_1__>* fifo) {
  return !fifo->empty();
}
extern "C" bool fpga_fifo_exist_1(hls::stream<__cosim_T_1__>* fifo) {
  return fifo->exist();
}
extern "C" bool fpga_direct_valid_1(hls::directio<__cosim_T_1__, 0>* direct) {
  return direct->valid();
}
extern "C" void fpga_direct_load_1(__cosim_T_1__* val, hls::directio<__cosim_T_1__, 0>* direct) {
  *val = direct->read();
}
extern "C" void fpga_direct_store_1(__cosim_T_1__* val, hls::directio<__cosim_T_1__, 0>* direct) {
  direct->write(*val);
}
struct __cosim_T_4__ {char data[4];};
extern "C" void fpga_fifo_push_4(__cosim_T_4__* val, hls::stream<__cosim_T_4__>* fifo) {
  fifo->write(*val);
}
extern "C" void fpga_fifo_pop_4(__cosim_T_4__* val, hls::stream<__cosim_T_4__>* fifo) {
  *val = fifo->read();
}
extern "C" bool fpga_fifo_not_empty_4(hls::stream<__cosim_T_4__>* fifo) {
  return !fifo->empty();
}
extern "C" bool fpga_fifo_exist_4(hls::stream<__cosim_T_4__>* fifo) {
  return fifo->exist();
}
extern "C" bool fpga_direct_valid_4(hls::directio<__cosim_T_4__, 0>* direct) {
  return direct->valid();
}
extern "C" void fpga_direct_load_4(__cosim_T_4__* val, hls::directio<__cosim_T_4__, 0>* direct) {
  *val = direct->read();
}
extern "C" void fpga_direct_store_4(__cosim_T_4__* val, hls::directio<__cosim_T_4__, 0>* direct) {
  direct->write(*val);
}
struct __cosim_T_12__ {char data[12];};
extern "C" void fpga_fifo_push_12(__cosim_T_12__* val, hls::stream<__cosim_T_12__>* fifo) {
  fifo->write(*val);
}
extern "C" void fpga_fifo_pop_12(__cosim_T_12__* val, hls::stream<__cosim_T_12__>* fifo) {
  *val = fifo->read();
}
extern "C" bool fpga_fifo_not_empty_12(hls::stream<__cosim_T_12__>* fifo) {
  return !fifo->empty();
}
extern "C" bool fpga_fifo_exist_12(hls::stream<__cosim_T_12__>* fifo) {
  return fifo->exist();
}
extern "C" bool fpga_direct_valid_12(hls::directio<__cosim_T_12__, 0>* direct) {
  return direct->valid();
}
extern "C" void fpga_direct_load_12(__cosim_T_12__* val, hls::directio<__cosim_T_12__, 0>* direct) {
  *val = direct->read();
}
extern "C" void fpga_direct_store_12(__cosim_T_12__* val, hls::directio<__cosim_T_12__, 0>* direct) {
  direct->write(*val);
}
