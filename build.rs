use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let lib_dir = PathBuf::from(&manifest_dir)
        .join("target/onnxruntime/lib");
    
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=target/onnxruntime/lib");
    
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-search=native={}", lib_dir.join("_deps/abseil_cpp-build/absl/base").display());
    println!("cargo:rustc-link-search=native={}", lib_dir.join("_deps/abseil_cpp-build/absl/container").display());
    println!("cargo:rustc-link-search=native={}", lib_dir.join("_deps/abseil_cpp-build/absl/crc").display());
    println!("cargo:rustc-link-search=native={}", lib_dir.join("_deps/abseil_cpp-build/absl/debugging").display());
    println!("cargo:rustc-link-search=native={}", lib_dir.join("_deps/abseil_cpp-build/absl/flags").display());
    println!("cargo:rustc-link-search=native={}", lib_dir.join("_deps/abseil_cpp-build/absl/hash").display());
    println!("cargo:rustc-link-search=native={}", lib_dir.join("_deps/abseil_cpp-build/absl/log").display());
    println!("cargo:rustc-link-search=native={}", lib_dir.join("_deps/abseil_cpp-build/absl/numeric").display());
    println!("cargo:rustc-link-search=native={}", lib_dir.join("_deps/abseil_cpp-build/absl/profiling").display());
    println!("cargo:rustc-link-search=native={}", lib_dir.join("_deps/abseil_cpp-build/absl/strings").display());
    println!("cargo:rustc-link-search=native={}", lib_dir.join("_deps/abseil_cpp-build/absl/synchronization").display());
    println!("cargo:rustc-link-search=native={}", lib_dir.join("_deps/abseil_cpp-build/absl/time").display());
    println!("cargo:rustc-link-search=native={}", lib_dir.join("_deps/abseil_cpp-build/absl/types").display());
    println!("cargo:rustc-link-search=native={}", lib_dir.join("_deps/protobuf-build").display());
    println!("cargo:rustc-link-search=native={}", lib_dir.join("_deps/pytorch_cpuinfo-build").display());
    println!("cargo:rustc-link-search=native={}", lib_dir.join("_deps/re2-build").display());
    
    let onnxruntime_libs = [
        "onnxruntime_providers",
        "onnxruntime_session", 
        "onnxruntime_optimizer",
        "onnxruntime_framework",
        "onnxruntime_graph",
        "onnxruntime_util",
        "onnxruntime_common",
        "onnxruntime_mlas",
        "onnxruntime_flatbuffers",
        "onnxruntime_lora",
        "onnx",
        "onnx_proto"
    ];
    
    for lib in &onnxruntime_libs {
        println!("cargo:rustc-link-lib=static={}", lib);
    }

    let absl_libs = [
        "absl_raw_hash_set",
        "absl_hashtablez_sampler",
        "absl_hash",
        "absl_city",
        "absl_low_level_hash",
        "absl_synchronization",
        "absl_graphcycles_internal",
        "absl_kernel_timeout_internal",
        "absl_time",
        "absl_civil_time",
        "absl_time_zone",
        "absl_strings",
        "absl_strings_internal",
        "absl_string_view",
        "absl_str_format_internal",
        "absl_cord",
        "absl_cordz_info",
        "absl_cord_internal",
        "absl_cordz_functions",
        "absl_cordz_handle",
        "absl_crc_cord_state",
        "absl_crc_cpu_detect",
        "absl_crc_internal",
        "absl_crc32c",
        "absl_int128",
        "absl_exponential_biased",
        "absl_base",
        "absl_spinlock_wait",
        "absl_malloc_internal",
        "absl_throw_delegate",
        "absl_raw_logging_internal",
        "absl_log_severity",
        "absl_strerror",
        "absl_log_internal_message",
        "absl_log_internal_check_op",
        "absl_log_internal_conditions",
        "absl_log_internal_format",
        "absl_log_internal_globals",
        "absl_log_internal_proto",
        "absl_log_internal_nullguard",
        "absl_log_internal_log_sink_set",
        "absl_log_sink",
        "absl_log_entry",
        "absl_log_globals",
        "absl_vlog_config_internal",
        "absl_log_internal_fnmatch",
        "absl_flags_commandlineflag",
        "absl_flags_commandlineflag_internal",
        "absl_flags_config",
        "absl_flags_internal",
        "absl_flags_marshalling",
        "absl_flags_private_handle_accessor",
        "absl_flags_program_name",
        "absl_flags_reflection",
        "absl_flags_usage",
        "absl_flags_usage_internal",
        "absl_flags_parse",
        "absl_symbolize",
        "absl_examine_stack",
        "absl_stacktrace",
        "absl_debugging_internal",
        "absl_demangle_internal",
        "absl_demangle_rust",
        "absl_decode_rust_punycode",
        "absl_utf8_for_code_point",
        "absl_failure_signal_handler",
        "absl_bad_any_cast_impl",
        "absl_bad_optional_access",
        "absl_bad_variant_access"
    ];
    
    for lib in &absl_libs {
        println!("cargo:rustc-link-lib=static={}", lib);
    }
    
    println!("cargo:rustc-link-lib=static=protobuf");
    println!("cargo:rustc-link-lib=static=cpuinfo");
    println!("cargo:rustc-link-lib=static=re2");
    
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=CoreFoundation");
}
