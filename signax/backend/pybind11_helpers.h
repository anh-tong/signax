#ifndef SIGNAX_PYBIND11_HELPER_H_
#define SIGNAX_PYBIND11_HELPER_H_

#include <cstdint>
#include <stdexcept>
#include <string.h>
#include <type_traits>
#include <pybind11/pybind11.h>

/*
This code is from
https://github.com/google/jax/blob/main/jaxlib/kernel_helpers.h
https://github.com/google/jax/blob/main/jaxlib/kernel_pybind11_helpers.h
*/

namespace signax 
{
    // This simply memory copy but with type cast
    template <class To, class From>
    typename std::enable_if<sizeof(To) == sizeof(From) && std::is_trivially_copyable<From>::value && std::is_trivially_copyable<To>::value, To>::type
    bit_cast(const From &src) noexcept
    {
        static_assert(
            std::is_trivially_constructible<To>::value,
            "Destination type should be trivially constructible");

        To dst;
        memcpy(&dst, &src, sizeof(To));
        return dst;
    }

    // To string function
    template <typename T>
    std::string PackDescriptorAsString(const T &descriptor)
    {
        return std::string(bit_cast<const char *>(&descriptor), sizeof(T));
    }

    // convert char to object type T
    // what is the `opaque` parameter here? This is one of the parameter of JAX
    // Python JAX will call `xla_client.ops.CustomCallWithLayout` with parameter `opaque`
    template <typename T>
    const T *UnpackDescriptor(const char *opaque, std::size_t opaque_len)
    {
        if (opaque_len != sizeof(T))
        {
            throw std::runtime_error("Invalid opaque object size");
        }

        return bit_cast<const T *>(opaque);
    }

    // convert descriptor to bytes
    template<typename T>
    pybind11::bytes PackDescriptor(const T &descriptor)
    {
        return pybind11::bytes(PackDescriptorAsString(descriptor));
    }

    // Encapsulate a C function to Python so that we can use it in Python
    template <typename T>
    pybind11::capsule EncapsulateFunction(T *fn)
    {
        return pybind11::capsule(bit_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
    }
}

#endif /* SIGNAX_PYBIND11_HELPER_H_ */
