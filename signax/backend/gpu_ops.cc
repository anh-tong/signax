#include "pybind11_helpers.h"

namespace signax{

    void gpu_signature_f32(cudaStream_t stream, void **buffer, const char* opaque, std::size_t opaque_len){
        // TODO: not implement
    }

    void gpu_signature_f64(cudaStream_t stream, void **buffer, const char* opaque, std::size_t opaque_len){
        // TODO: not implement
    }

    pybind11::dict Registrations()
    {
        pybind11::dict dict;
        dict["gpu_signature_f32"] = EncapsulateFunction(gpu_signature_f32);
        dict["gpu_signature_f64"] = EncapsulateFunction(gpu_signature_f64);
        return dict;
    }

    PYBIND11_MODULE(gpu_ops, m)
    {
        m.def("registrations", &Registrations);
    }

}
