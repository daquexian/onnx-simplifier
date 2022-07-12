#pragma once

#include "cxxopts.hpp"

class OnnxsimOption {
 public:
  OnnxsimOption() = default;
  OnnxsimOption(int argc, char** argv) { Parse(argc, argv); }
  ~OnnxsimOption() = default;

  void Parse(int argc, char** argv);

  template <typename T>
  T Get(const std::string& key) const {
    T value = options_[key].as<T>();
    return value;
  }

 private:
  cxxopts::ParseResult options_;
};