#include <iostream>

#include "ck_tile/core/container/multi_index.hpp"
#include "ck_tile/core/container/tuple.hpp"

int main() {
 
  auto arg = ck_tile::tuple<int, int, int, int>(11, 2, 5, 1);	
  // arg must have a static size() method
  // some of the containers in ck_tile/core/container that is.
  auto multi_index = ck_tile::to_multi_index(arg);
  for (int i = 0; i < multi_index.size(); ++i) { std::cout << i << ": " << multi_index.at(i) << ", " << std::endl; }

  return 0;
}

