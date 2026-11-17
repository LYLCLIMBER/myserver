#include <iostream>
#include <llama.h>

int main ()
{
  std::cout << "Test llama import\n";
  llama_backend_init();
  llama_backend_free();
}
