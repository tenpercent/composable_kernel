#include <iostream>

#include "ck_tile/core/container/multi_index.hpp"

int main() {
  
  auto zero_multi_index = ck_tile::make_zero_multi_index<8>();
  for (int i = 0; i < zero_multi_index.size(); ++i) { std::cout << i << ": " << zero_multi_index.at(i) << ", " << std::endl; }

  return 0;
}

