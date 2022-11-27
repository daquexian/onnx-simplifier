#include "onnxsim_option.h"
#include <iostream>

void OnnxsimOption::Parse(int argc, char** argv) {
  cxxopts::Options cxx_options("onnxsim", "Simplify your ONNX model");

  // clang-format off
  cxx_options.add_options()
  ("h,help",              "Print help")
  ("i,input-model",       "Input onnx model filename. This argument is required.",   cxxopts::value<std::string>())
  ("o,output-model",      "Output onnx model filename. This argument is required.",  cxxopts::value<std::string>())
  ("no-opt",              "No optimization",             cxxopts::value<bool>()->default_value("false"))
  ("no-sim",              "No simplification",           cxxopts::value<bool>()->default_value("false"))
  ("no-shape-inference",  "No shape inference",          cxxopts::value<bool>()->default_value("false"))
  ;
  // clang-format on

  try {
    options_ = cxx_options.parse(argc, argv);
  } catch (cxxopts::OptionParseException cxxopts_exception) {
    std::cout << "[Error] Can not parse your options" << std::endl;
    std::cout << cxx_options.help() << std::endl;
    exit(1);
  }

  if (options_.count("help")) {
    std::cout << cxx_options.help() << std::endl;
    exit(0);
  }
  if (!options_.count("input-model") || !options_.count("output-model")) {
    std::cout << cxx_options.help() << std::endl;
    exit(1);
  }

  return;
}
