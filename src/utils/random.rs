//! Fast 16-bit pseudo random number generator.

// Based on MIT-licensed code (c) 2012 by Olivier Gillet (ol.gillet@gmail.com)

use core::sync::atomic::{AtomicU32, Ordering};

const MULTIPLIER: u32 = 1664525;
const INCREMENT: u32 = 1013904223;

/// Default seed for owned generators and the legacy process-wide generator.
pub const DEFAULT_SEED: u32 = 0x21;

static RNG_STATE: AtomicU32 = AtomicU32::new(DEFAULT_SEED);

#[inline]
fn next_state(state: u32) -> u32 {
    state.wrapping_mul(MULTIPLIER).wrapping_add(INCREMENT)
}

#[inline]
fn state() -> u32 {
    RNG_STATE.load(Ordering::Relaxed)
}

#[inline]
pub fn seed(seed: u32) {
    RNG_STATE.store(seed, Ordering::Relaxed);
}

#[inline]
pub fn get_word() -> u32 {
    RNG_STATE.store(
        next_state(RNG_STATE.load(Ordering::Relaxed)),
        Ordering::Relaxed,
    );
    state()
}

#[inline]
pub fn get_sample() -> i16 {
    (get_word() >> 16) as i16
}

#[inline]
pub fn get_float() -> f32 {
    get_word() as f32 / 4294967296.0
}

/// Owned pseudo random number generator.
///
/// Cloning copies the current state, so the clone continues with exactly the
/// same sequence. [`Default`] starts at [`DEFAULT_SEED`].
#[derive(Debug, Clone)]
pub struct Rng {
    state: u32,
}

impl Rng {
    /// Creates a generator starting at `seed`.
    pub const fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    /// Restarts the sequence at `seed`.
    pub fn seed(&mut self, seed: u32) {
        self.state = seed;
    }

    /// Returns the current state.
    pub fn state(&self) -> u32 {
        self.state
    }

    /// Advances the generator and returns the next word.
    #[inline]
    pub fn get_word(&mut self) -> u32 {
        self.state = next_state(self.state);
        self.state
    }

    /// Advances the generator and returns the high 16 bits as a sample.
    #[inline]
    pub fn get_sample(&mut self) -> i16 {
        (self.get_word() >> 16) as i16
    }

    /// Advances the generator and returns a value in the range `[0.0, 1.0)`.
    #[inline]
    pub fn get_float(&mut self) -> f32 {
        self.get_word() as f32 / 4294967296.0
    }
}

impl Default for Rng {
    fn default() -> Self {
        Self::new(DEFAULT_SEED)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_and_process_wide_generators_produce_the_same_sequence() {
        const SEED: u32 = 0xdead_beef;

        let mut rng = Rng::new(SEED);
        seed(SEED);

        for _ in 0..32 {
            assert_eq!(rng.get_word(), get_word());
        }
    }

    #[test]
    fn default_starts_at_default_seed() {
        assert_eq!(Rng::default().state(), DEFAULT_SEED);
    }

    #[test]
    fn seed_restarts_the_sequence() {
        const SEED: u32 = 0x1234_5678;

        let mut rng = Rng::new(SEED);
        let first_word = rng.get_word();
        let _ = rng.get_word();
        rng.seed(SEED);

        assert_eq!(rng.get_word(), first_word);
    }

    #[test]
    fn state_round_trips_through_new() {
        let mut rng = Rng::new(0x1234_5678);
        let _ = rng.get_word();
        let mut resumed = Rng::new(rng.state());

        assert_eq!(resumed.get_word(), rng.get_word());
    }
}
