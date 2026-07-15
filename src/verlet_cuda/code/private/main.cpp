#include "klvk/error_handling.hpp"
#include "verlet_cuda_app.hpp"

void Main()
{
    verlet::VerletCudaApp app;
    app.Run();
}

int main()
{
    klvk::ErrorHandling::InvokeAndCatchAll(Main);
    return 0;
}
