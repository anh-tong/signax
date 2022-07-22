
#include "pybind11_helpers.h"

namespace signax
{
    template <typename T>
    void cpu_signature(void *out, const void **in){
        // TODO: implement this
    }

    pybind11::dict Registrations()
    {
        pybind11::dict dict;
        dict["cpu_signature_f32"] = EncapsulateFunction(cpu_signature<float>);
        dict["cpu_signature_f64"] = EncapsulateFunction(cpu_signature<double>);
        return dict;
    }

    PYBIND11_MODULE(cpu_ops, m) { m.def("registrations", &Registrations); }
}
