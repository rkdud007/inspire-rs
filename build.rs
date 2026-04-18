use std::process::Command;

#[allow(dead_code)]
fn compile_cuda(cuda_path: &str, out_dir: &str, src: &str, name: &str) {
    println!("cargo:rerun-if-changed={}", src);

    let obj_path = format!("{}/{}.o", out_dir, name);
    let lib_path = format!("{}/lib{}.a", out_dir, name);

    let nvcc_status = Command::new(format!("{}/bin/nvcc", cuda_path))
        .args([
            "-O3",
            "-c",
            src,
            "-o",
            &obj_path,
            "--compiler-options",
            "-fPIC",
            "-arch=sm_90",
            "--default-stream",
            "per-thread",
            "-Isrc/cuda", // for common.h
        ])
        .status()
        .unwrap_or_else(|e| panic!("Failed to run nvcc for {}: {}", src, e));

    if !nvcc_status.success() {
        panic!("nvcc failed to compile {}", src);
    }

    let ar_status = Command::new("ar")
        .args(["rcs", &lib_path, &obj_path])
        .status()
        .expect("Failed to run ar");

    if !ar_status.success() {
        panic!("ar failed to create {}", lib_path);
    }

    println!("cargo:rustc-link-lib=static={}", name);
}

fn main() {
    // Compile C++ matmul (portable)
    println!("cargo:rerun-if-changed=src/matmul.cpp");
    cc::Build::new()
        .cpp(true)
        .file("src/matmul.cpp")
        .flag("-O3")
        .flag("-march=native")
        .flag("-std=c++11")
        .compile("matmul");

    // CUDA compilation only when gpu feature is enabled
    if cfg!(feature = "gpu") {
        let cuda_path =
            std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
        let out_dir = std::env::var("OUT_DIR").unwrap();

        // Compile CUDA kernels
        println!("cargo:rerun-if-changed=src/cuda/common.h");
        compile_cuda(
            &cuda_path,
            &out_dir,
            "src/cuda/encode_kernel.cu",
            "encode_kernel",
        );
        compile_cuda(
            &cuda_path,
            &out_dir,
            "src/cuda/packing_kernel.cu",
            "packing_kernel",
        );
        compile_cuda(
            &cuda_path,
            &out_dir,
            "src/cuda/gemv_kernel.cu",
            "gemv_kernel",
        );
        compile_cuda(
            &cuda_path,
            &out_dir,
            "src/cuda/packing_online_kernel.cu",
            "packing_online_kernel",
        );
        compile_cuda(
            &cuda_path,
            &out_dir,
            "src/cuda/rotation_kernel.cu",
            "rotation_kernel",
        );
        compile_cuda(
            &cuda_path,
            &out_dir,
            "src/cuda/hint_kernel.cu",
            "hint_kernel",
        );
        compile_cuda(
            &cuda_path,
            &out_dir,
            "src/cuda/prep_pack_kernel.cu",
            "prep_pack_kernel",
        );
        compile_cuda(
            &cuda_path,
            &out_dir,
            "src/cuda/collapse_kernel.cu",
            "collapse_kernel",
        );

        // Link CUDA libraries
        println!("cargo:rustc-link-search=native={}", out_dir);
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}
