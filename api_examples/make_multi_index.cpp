#include <iostream>

#include "ck_tile/core/container/multi_index.hpp"

int main() {
  
  auto multi_index = ck_tile::make_multi_index(1, 5, 2, 8, 3);
  for (int i = 0; i < multi_index.size(); ++i) { std::cout << i << ": " << multi_index.at(i) << ", " << std::endl; }

  return 0;
}

