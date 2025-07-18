MIN_FIB_LOG=18 MAX_FIB_LOG=20 RUSTFLAGS="-C target-cpu=native -C opt-level=3" \
    cargo test test_wide_fib_prove_with_blake_icicle --release --features icicle,parallel -- --nocapture
