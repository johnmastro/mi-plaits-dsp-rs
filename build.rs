fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    println!(
        "cargo:rustc-env=GOLDEN_ENCODED_RUSTFLAGS={}",
        std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default()
    );
}
