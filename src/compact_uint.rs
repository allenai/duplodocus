use std::fmt;
use bytemuck::{Pod, Zeroable};

/// A trait for compact unsigned integer types that can be used as suffix array indices.
/// 
/// This trait allows the suffix array implementation to work with different integer sizes,
/// trading off between memory usage and maximum addressable text size.
pub trait CompactUint: 
    Copy + Clone + Ord + Eq + fmt::Debug + Send + Sync + 'static 
{
    /// Convert from a u64 value.
    /// 
    /// # Panics
    /// May panic if val > Self::max_value()
    fn from_u64(val: u64) -> Self;
    
    /// Convert to a u64 value.
    fn to_u64(self) -> u64;
    
    /// The maximum value this type can represent.
    fn max_value() -> u64;
    
    /// Return zero.
    fn zero() -> Self;
    
    /// Return the maximum value as Self.
    fn max_I() -> Self;

    /// The size of this type in bytes.
    const BYTE_SIZE: usize;
}

// Implementation for u32
impl CompactUint for u32 {
    #[inline(always)]
    fn from_u64(val: u64) -> Self {
        debug_assert!(val <= Self::max_value().into());
        val as u32
    }
    
    #[inline(always)]
    fn to_u64(self) -> u64 {
        self as u64
    }
    
    #[inline(always)]
    fn max_value() -> u64 {
        u32::MAX as u64
    }
    
    #[inline(always)]
    fn zero() -> Self {
        0
    }
    
    #[inline(always)]
    fn max_I() -> Self {
        u32::MAX
    }
    
    const BYTE_SIZE: usize = 4;
}

// Implementation for u64
impl CompactUint for u64 {
    #[inline(always)]
    fn from_u64(val: u64) -> Self {
        val
    }
    
    #[inline(always)]
    fn to_u64(self) -> u64 {
        self
    }
    
    #[inline(always)]
    fn max_value() -> u64 {
        u64::MAX
    }
    
    #[inline(always)]
    fn zero() -> Self {
        0
    }
    
    #[inline(always)]
    fn max_I() -> Self {
        u64::MAX
    }
    
    const BYTE_SIZE: usize = 8;
}

/// A 40-bit unsigned integer type.
/// 
/// This type can represent values from 0 to 2^40-1 (1,099,511,627,775),
/// which is sufficient for indexing into 1TB of text data.
/// 
/// Uses 5 bytes of storage compared to 8 bytes for u64, providing a 37.5%
/// memory reduction for suffix arrays on large datasets.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
#[repr(C)] // Align to 8 bytes to help with cache line efficiency
pub struct U40([u8; 5]);

unsafe impl Zeroable for U40 {}
unsafe impl Pod for U40 {}

impl U40 {
    /// Maximum value that can be stored in a U40 (2^40 - 1)
    pub const MAX: u64 = (1u64 << 40) - 1;
    
    /// Create a new U40 from a u64, panicking if the value is too large
    #[inline]
    pub fn new(val: u64) -> Self {
        assert!(val <= Self::MAX, "Value {} exceeds U40::MAX ({})", val, Self::MAX);
        Self::from_u64(val)
    }
    
    /// Create a new U40 from a u64, returning None if the value is too large
    #[inline]
    pub fn try_new(val: u64) -> Option<Self> {
        if val <= Self::MAX {
            Some(Self::from_u64(val))
        } else {
            None
        }
    }
}

impl CompactUint for U40 {
    #[inline(always)]
    fn from_u64(val: u64) -> Self {
        debug_assert!(val <= Self::max_value());
        U40([
            val as u8,
            (val >> 8) as u8,
            (val >> 16) as u8,
            (val >> 24) as u8,
            (val >> 32) as u8,
        ])
    }
    
    #[inline(always)]
    fn to_u64(self) -> u64 {
        (self.0[0] as u64)
            | ((self.0[1] as u64) << 8)
            | ((self.0[2] as u64) << 16)
            | ((self.0[3] as u64) << 24)
            | ((self.0[4] as u64) << 32)
    }
    
    #[inline(always)]
    fn max_value() -> u64 {
        Self::MAX
    }
    
    #[inline(always)]
    fn zero() -> Self {
        U40([0, 0, 0, 0, 0])
    }
    
    #[inline(always)]
    fn max_I() -> Self {
        U40([0xFF, 0xFF, 0xFF, 0xFF, 0xFF])
    }
    
    const BYTE_SIZE: usize = 5;
}

impl fmt::Debug for U40 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "U40({})", self.to_u64())
    }
}

impl fmt::Display for U40 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_u64())
    }
}


#[inline(always)]
pub unsafe fn read_compact_uint_unchecked<I: CompactUint>(bytes: &[u8]) -> u64 {
    match I::BYTE_SIZE {
        4 => {
            let ptr = bytes.as_ptr() as *const u32;
            (*ptr).to_le() as u64
        }
        5 => {
            let ptr = bytes.as_ptr();
            (*ptr as u64)
                | ((*ptr.add(1) as u64) << 8)
                | ((*ptr.add(2) as u64) << 16)
                | ((*ptr.add(3) as u64) << 24)
                | ((*ptr.add(4) as u64) << 32)
        }
        8 => {
            let ptr = bytes.as_ptr() as *const u64;
            (*ptr).to_le()
        }
        _ => unreachable!(),
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_u40_basic() {
        let zero = U40::zero();
        assert_eq!(zero.to_u64(), 0);
        
        let max = U40::max_value();
        assert_eq!(max.to_u64(), U40::MAX);
        
        let mid = U40::from_u64(1_000_000_000);
        assert_eq!(mid.to_u64(), 1_000_000_000);
    }
    
    #[test]
    fn test_u40_roundtrip() {
        let test_values = vec![
            0,
            1,
            255,
            256,
            65535,
            65536,
            1_000_000,
            1_000_000_000,
            U40::MAX - 1,
            U40::MAX,
        ];
        
        for val in test_values {
            let u40 = U40::from_u64(val);
            assert_eq!(u40.to_u64(), val, "Roundtrip failed for {}", val);
        }
    }
    
    #[test]
    fn test_u40_ordering() {
        let a = U40::from_u64(100);
        let b = U40::from_u64(200);
        let c = U40::from_u64(100);
        
        assert!(a < b);
        assert!(b > a);
        assert_eq!(a, c);
        assert!(a <= c);
        assert!(a >= c);
    }
    
    #[test]
    #[should_panic]
    fn test_u40_overflow_new() {
        U40::new(U40::MAX + 1);
    }
    
    #[test]
    fn test_u40_try_new() {
        assert!(U40::try_new(U40::MAX).is_some());
        assert!(U40::try_new(U40::MAX + 1).is_none());
    }
    
    #[test]
    fn test_compact_uint_trait() {
        // Test that all implementations work correctly
        fn test_impl<I: CompactUint>() {
            let zero = I::zero();
            assert_eq!(zero.to_u64(), 0);
            
            let hundred = I::from_u64(100);
            assert_eq!(hundred.to_u64(), 100);
            
            assert!(I::max_value() > 0);
        }
        
        test_impl::<u32>();
        test_impl::<u64>();
        test_impl::<U40>();
    }
}