#include "cuda_verlet_app.hpp"
#include "klgl/error_handling.hpp"

void Main()
{
    verlet::VerletCudaApp app;
    app.Run();
}

int main()
{
    klgl::ErrorHandling::InvokeAndCatchAll(Main);
    return 0;
}
