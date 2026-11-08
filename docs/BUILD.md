# Build and installation

VTLC targets GCC and Clang with C++20 on x86-64 Linux. CMake 3.16 or newer and POSIX threads are required. Zstandard headers and library are required by the default configuration; the codec can also be built without Zstandard. The benchmark executable uses Linux/POSIX process APIs. Other systems and compilers have not received the same validation.

## Release build

From the project root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

For example, Debian-based distributions provide the dependencies through packages such as `cmake`, a C++ compiler, and `libzstd-dev`. Package names vary by distribution. CMake's `find_library` locates `zstd`; the compiler must also find `zstd.h` on its include path.

To compile without the Zstandard dependency:

```sh
cmake -S . -B build-nozstd -DCMAKE_BUILD_TYPE=Release -DVTLC_ZSTD=OFF
cmake --build build-nozstd -j
ctest --test-dir build-nozstd --output-on-failure
```

Such a build can encode and decode its own non-Zstandard modes. It rejects tiles encoded with Zstandard by another build. The Zstandard comparison benchmark and its codec-specific test are omitted when this option is off.

## Install and link

```sh
cmake --install build --prefix "$PWD/install"
c++ -std=c++20 -I install/include examples/basic.cpp \
    -L install/lib -Wl,-rpath,"$PWD/install/lib" -lvtlc -o build/basic-example
./build/basic-example
cc -I install/include examples/basic.c \
   -L install/lib -Wl,-rpath,"$PWD/install/lib" -lvtlc -o build/c-example
./build/c-example
```

The install contains `lib/libvtlc.so` and headers under `include/vtlc`. Consumers can instead arrange for the shared library to be found by the system loader. The C example is compiled as C and links to the C++ implementation through the shared library.

## ISA and compiler settings

The build does not enable AVX2 or AVX-512 globally. Target-specific functions are selected at runtime after compiler-provided CPU feature detection; requesting an unavailable backend fails. `Auto` prefers AVX-512F, then AVX2+FMA, AVX2, and scalar. AVX-512F code avoids assuming AVX-512BW, DQ or VL. Bitshuffle and other byte operations use AVX2 only when it is independently available.

The library is built with `-fno-fast-math`, `-frounding-math`, and `-ffp-contract=off`. Fused prediction explicitly calls a fused operation and controls x86 MXCSR for that operation. Do not remove those flags or silently change the predictor semantics when distributing compatible builds. The AVX-512 implementation has not been executed on the recorded development machine.

## Build options

| CMake option | Default | Effect |
|---|---:|---|
| `VTLC_ZSTD` | `ON` | Enable Zstandard tile modes and comparison benchmark. |
| `VTLC_SANITIZE` | `OFF` | Enable AddressSanitizer and UndefinedBehaviorSanitizer. |
| `VTLC_FUZZ` | `OFF` | Build the decoder libFuzzer target; requires Clang. |

See [TESTING.md](TESTING.md) for sanitizer and fuzzing commands and [BENCHMARKS.md](BENCHMARKS.md) for benchmark interpretation.
