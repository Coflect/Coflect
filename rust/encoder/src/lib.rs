//! Rust encoder placeholder crate for Coflect performance paths.

/// Placeholder API. Replace with real encode/serialize routines.
pub fn encode_stub(input: &[u8]) -> Vec<u8> {
    input.to_vec()
}

#[cfg(test)]
mod tests {
    use super::encode_stub;

    #[test]
    fn round_trip_stub() {
        let data = [1_u8, 2_u8, 3_u8];
        assert_eq!(encode_stub(&data), data);
    }
}
