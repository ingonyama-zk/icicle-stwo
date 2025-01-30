use std::fmt::Display;
use std::str::FromStr;

pub fn get_env_var<T: Display + FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .unwrap_or_else(|_| default.to_string())
        .parse()
        .unwrap_or(default)
}
