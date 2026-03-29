// Build script for voxtral-tts-rs.
//
// - When `tch-backend` is enabled: embeds libtorch RPATH into the binary
// - When `mlx` is enabled (macOS only): builds mlx-c from submodule via CMake,
//   links static libraries and macOS frameworks, embeds RPATH

fn main() {
    #[cfg(feature = "tch-backend")]
    setup_libtorch_rpath();

    #[cfg(feature = "mlx")]
    build_mlx();
}

/// Embed libtorch RPATH so the binary finds shared libs at runtime without LD_LIBRARY_PATH.
///
/// Two RPATHs are embedded:
/// 1. Relative to the binary (`$ORIGIN/../libtorch/lib`) — for deployed layouts
/// 2. Absolute from `$LIBTORCH` env var (if set) — for the build-time location
#[cfg(feature = "tch-backend")]
fn setup_libtorch_rpath() {
    // Always embed a relative RPATH so deployed binaries work without LD_LIBRARY_PATH
    #[cfg(target_os = "linux")]
    {
        // Flat layout: binary and libtorch/ in the same directory (release zip)
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/libtorch/lib");
        // Nested layout: binary in bin/ subdirectory
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../libtorch/lib");
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib");
    }

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../libtorch/lib");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../lib");
    }

    // Also embed the absolute build-time path so tests/CI work in-tree
    if let Ok(libtorch_path) = std::env::var("LIBTORCH") {
        let lib_dir = format!("{}/lib", libtorch_path);
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir);
    }
}

/// Build the mlx-c library from the git submodule and emit linker directives.
#[cfg(feature = "mlx")]
fn build_mlx() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    if target_os != "macos" {
        panic!("The `mlx` feature is only supported on macOS. Current target OS: {target_os}");
    }
    if target_arch != "aarch64" {
        eprintln!(
            "Warning: MLX is optimized for Apple Silicon (aarch64). \
             Current target arch: {target_arch}. Metal GPU acceleration may not be available."
        );
    }

    let mlx_c_dir = std::path::PathBuf::from("mlx-c");
    if !mlx_c_dir.join("CMakeLists.txt").exists() {
        panic!(
            "mlx-c submodule not found. Please run:\n\
             \n\
             git submodule update --init --recursive\n\
             \n\
             to clone the mlx-c dependency."
        );
    }

    // Build mlx-c via CMake
    let dst = cmake::Config::new(&mlx_c_dir)
        .define("MLX_BUILD_TESTS", "OFF")
        .define("MLX_BUILD_EXAMPLES", "OFF")
        .define("MLX_BUILD_BENCHMARKS", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .build();

    // Link paths
    let lib_dir = dst.join("lib");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Also check lib64 (some CMake configs use this)
    let lib64_dir = dst.join("lib64");
    if lib64_dir.exists() {
        println!("cargo:rustc-link-search=native={}", lib64_dir.display());
    }

    // Link mlx-c and mlx static libraries
    println!("cargo:rustc-link-lib=static=mlxc");
    println!("cargo:rustc-link-lib=static=mlx");

    // Link macOS system frameworks required by MLX
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");

    // Link C++ standard library
    println!("cargo:rustc-link-lib=c++");

    // Embed RPATH for MLX libraries
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    // Rerun if mlx-c sources change
    println!("cargo:rerun-if-changed=mlx-c/CMakeLists.txt");
    println!("cargo:rerun-if-changed=mlx-c/mlx/c/");
}
