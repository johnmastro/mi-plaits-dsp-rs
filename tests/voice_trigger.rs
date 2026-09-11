//! Public-API tests for gate delay and explicit trigger behavior.

use mi_plaits_dsp::engine::TriggerState;
use mi_plaits_dsp::voice::{
    DEFAULT_TRIGGER_DELAY, IMMEDIATE_TRIGGER_DELAY, LEGACY_TRIGGER_DELAY, MAX_TRIGGER_DELAY,
    Modulations, Patch, Voice,
};

const SAMPLE_RATE: f32 = 48_000.0;
const BLOCK_SIZE: usize = 24;
const TEST_SEED: u32 = 0x1234_5678;

fn initialized_voice(trigger_delay_blocks: usize) -> Voice<'static> {
    let mut voice = Voice::new_with_trigger_delay(BLOCK_SIZE, SAMPLE_RATE, trigger_delay_blocks);
    voice.seed_rng(TEST_SEED);
    voice.init();
    voice
}

fn render(
    voice: &mut Voice<'static>,
    patch: &Patch,
    modulations: &Modulations,
) -> ([f32; BLOCK_SIZE], [f32; BLOCK_SIZE]) {
    let mut out = [0.0; BLOCK_SIZE];
    let mut aux = [0.0; BLOCK_SIZE];
    voice.render(patch, modulations, &mut out, &mut aux);
    (out, aux)
}

fn render_gate(voice: &mut Voice<'static>, trigger: f32, trigger_patched: bool) -> TriggerState {
    let modulations = Modulations {
        trigger,
        trigger_patched,
        ..Modulations::default()
    };
    render(voice, &Patch::default(), &modulations);
    voice.last_trigger()
}

#[test]
fn every_logical_trigger_delay_is_exact() {
    const RISE_AT: usize = 2;
    const FALL_AT: usize = 6;

    for delay in IMMEDIATE_TRIGGER_DELAY..=MAX_TRIGGER_DELAY {
        let mut voice = initialized_voice(delay);

        for block in 0..=(FALL_AT + delay + 1) {
            let trigger = if (RISE_AT..FALL_AT).contains(&block) {
                1.0
            } else {
                0.0
            };
            let actual = render_gate(&mut voice, trigger, true);
            let expected = if block < RISE_AT + delay {
                TriggerState::Low
            } else if block == RISE_AT + delay {
                TriggerState::RisingEdge
            } else if block < FALL_AT + delay {
                TriggerState::High
            } else {
                TriggerState::Low
            };

            assert_eq!(actual, expected, "delay {delay}, block {block}");
        }
    }
}

#[test]
fn legacy_startup_matches_upstream() {
    let mut voice = initialized_voice(LEGACY_TRIGGER_DELAY);

    for block in 0..LEGACY_TRIGGER_DELAY {
        assert_eq!(
            render_gate(&mut voice, 1.0, true),
            TriggerState::Low,
            "block {block}"
        );
    }
    assert_eq!(render_gate(&mut voice, 1.0, true), TriggerState::RisingEdge);
}

#[test]
fn immediate_startup_is_immediate() {
    let mut voice = initialized_voice(IMMEDIATE_TRIGGER_DELAY);

    assert_eq!(render_gate(&mut voice, 1.0, true), TriggerState::RisingEdge);
}

#[test]
fn old_constructor_remains_legacy() {
    let voice = Voice::new(BLOCK_SIZE, SAMPLE_RATE);

    assert_eq!(DEFAULT_TRIGGER_DELAY, LEGACY_TRIGGER_DELAY);
    assert_eq!(voice.trigger_delay(), LEGACY_TRIGGER_DELAY);
    assert_eq!(voice.last_trigger(), TriggerState::Low);
}

#[test]
fn invalid_trigger_delays_panic() {
    for delay in [MAX_TRIGGER_DELAY + 1, usize::MAX] {
        assert!(
            std::panic::catch_unwind(|| {
                Voice::new_with_trigger_delay(BLOCK_SIZE, SAMPLE_RATE, delay)
            })
            .is_err(),
            "delay {delay} did not panic"
        );
    }
}

#[test]
fn one_quantum_gate_gap_retriggers_at_every_delay() {
    for delay in IMMEDIATE_TRIGGER_DELAY..=MAX_TRIGGER_DELAY {
        let mut voice = initialized_voice(delay);
        let gap_at = delay + 3;
        let observed_low = gap_at + delay;
        let observed_rise = observed_low + 1;
        let mut states = Vec::new();

        for block in 0..=(observed_rise + 1) {
            let trigger = if block == gap_at { 0.0 } else { 1.0 };
            states.push(render_gate(&mut voice, trigger, true));
        }

        assert_eq!(states[delay], TriggerState::RisingEdge, "delay {delay}");
        assert_eq!(states[observed_low], TriggerState::Low, "delay {delay}");
        assert_eq!(
            states[observed_rise],
            TriggerState::RisingEdge,
            "delay {delay}"
        );
        assert_eq!(
            states[observed_rise + 1],
            TriggerState::High,
            "delay {delay}"
        );
    }
}

#[test]
fn trigger_hysteresis_preserves_strict_boundaries_and_hold_band() {
    let mut voice = initialized_voice(IMMEDIATE_TRIGGER_DELAY);
    let just_above_high = f32::from_bits(0.3_f32.to_bits() + 1);
    let just_below_low = f32::from_bits(0.1_f32.to_bits() - 1);

    assert_eq!(render_gate(&mut voice, 0.3, true), TriggerState::Low);
    assert_eq!(render_gate(&mut voice, 0.2, true), TriggerState::Low);
    assert_eq!(
        render_gate(&mut voice, just_above_high, true),
        TriggerState::RisingEdge
    );
    assert_eq!(render_gate(&mut voice, 0.3, true), TriggerState::High);
    assert_eq!(render_gate(&mut voice, 0.1, true), TriggerState::High);
    assert_eq!(render_gate(&mut voice, 0.2, true), TriggerState::High);
    assert_eq!(
        render_gate(&mut voice, just_below_low, true),
        TriggerState::Low
    );
    assert_eq!(render_gate(&mut voice, 0.1, true), TriggerState::Low);
}

#[test]
fn held_high_gate_can_be_explicitly_retriggered() {
    let delay = 3;
    let mut voice = initialized_voice(delay);

    for _ in 0..delay {
        assert_eq!(render_gate(&mut voice, 1.0, true), TriggerState::Low);
    }
    assert_eq!(render_gate(&mut voice, 1.0, true), TriggerState::RisingEdge);
    assert_eq!(render_gate(&mut voice, 1.0, true), TriggerState::High);

    voice.trigger();
    assert_eq!(render_gate(&mut voice, 1.0, true), TriggerState::RisingEdge);
    assert_eq!(render_gate(&mut voice, 1.0, true), TriggerState::High);
}

#[test]
fn explicit_trigger_with_low_gate_is_a_one_shot() {
    let mut voice = initialized_voice(LEGACY_TRIGGER_DELAY);

    voice.trigger();
    assert_eq!(render_gate(&mut voice, 0.0, true), TriggerState::RisingEdge);
    assert_eq!(render_gate(&mut voice, 0.0, true), TriggerState::Low);
}

#[test]
fn repeated_explicit_triggers_before_render_coalesce() {
    let mut voice = initialized_voice(LEGACY_TRIGGER_DELAY);

    voice.trigger();
    voice.trigger();
    assert_eq!(render_gate(&mut voice, 0.0, true), TriggerState::RisingEdge);
    for _ in 0..=LEGACY_TRIGGER_DELAY {
        assert_eq!(render_gate(&mut voice, 0.0, true), TriggerState::Low);
    }
}

#[test]
fn forced_rise_resynchronizes_a_delayed_high_gate() {
    let mut voice = initialized_voice(LEGACY_TRIGGER_DELAY);

    voice.trigger();
    assert_eq!(render_gate(&mut voice, 1.0, true), TriggerState::RisingEdge);
    for block in 1..=(LEGACY_TRIGGER_DELAY + 1) {
        assert_eq!(
            render_gate(&mut voice, 1.0, true),
            TriggerState::High,
            "block {block}"
        );
    }
}

#[test]
fn forced_falling_gate_is_low_on_the_following_render() {
    let mut voice = initialized_voice(LEGACY_TRIGGER_DELAY);

    for _ in 0..LEGACY_TRIGGER_DELAY {
        assert_eq!(render_gate(&mut voice, 1.0, true), TriggerState::Low);
    }
    assert_eq!(render_gate(&mut voice, 1.0, true), TriggerState::RisingEdge);
    assert_eq!(render_gate(&mut voice, 1.0, true), TriggerState::High);

    voice.trigger();
    assert_eq!(render_gate(&mut voice, 0.0, true), TriggerState::RisingEdge);
    for block in 1..=(LEGACY_TRIGGER_DELAY + 1) {
        assert_eq!(
            render_gate(&mut voice, 0.0, true),
            TriggerState::Low,
            "block after forced fall {block}"
        );
    }
}

#[test]
fn unrelated_rise_after_forced_one_shot_uses_configured_delay() {
    let mut voice = initialized_voice(LEGACY_TRIGGER_DELAY);

    voice.trigger();
    assert_eq!(render_gate(&mut voice, 0.0, true), TriggerState::RisingEdge);
    for _ in 0..=LEGACY_TRIGGER_DELAY {
        assert_eq!(render_gate(&mut voice, 0.0, true), TriggerState::Low);
    }

    for offset in 0..LEGACY_TRIGGER_DELAY {
        assert_eq!(
            render_gate(&mut voice, 1.0, true),
            TriggerState::Low,
            "offset {offset}"
        );
    }
    assert_eq!(render_gate(&mut voice, 1.0, true), TriggerState::RisingEdge);
}

#[test]
fn explicit_trigger_discards_an_in_flight_delayed_pulse() {
    let mut voice = initialized_voice(LEGACY_TRIGGER_DELAY);
    let mut states = Vec::new();

    states.push(render_gate(&mut voice, 1.0, true));
    states.push(render_gate(&mut voice, 0.0, true));
    voice.trigger();
    states.push(render_gate(&mut voice, 0.0, true));
    for _ in 0..=(LEGACY_TRIGGER_DELAY + 2) {
        states.push(render_gate(&mut voice, 0.0, true));
    }

    assert_eq!(states[2], TriggerState::RisingEdge);
    assert_eq!(
        states
            .iter()
            .filter(|state| **state == TriggerState::RisingEdge)
            .count(),
        1
    );
    assert!(states[3..].iter().all(|state| *state == TriggerState::Low));
}

#[test]
fn unpatched_explicit_trigger_is_consumed_and_resynchronizes_history() {
    let delay = 3;
    let mut voice = initialized_voice(delay);

    voice.trigger();
    assert_eq!(render_gate(&mut voice, 1.0, false), TriggerState::Unpatched);
    assert_eq!(render_gate(&mut voice, 1.0, true), TriggerState::High);

    for offset in 0..delay {
        assert_eq!(
            render_gate(&mut voice, 0.0, true),
            TriggerState::High,
            "fall offset {offset}"
        );
    }
    assert_eq!(render_gate(&mut voice, 0.0, true), TriggerState::Low);

    for offset in 0..delay {
        assert_eq!(
            render_gate(&mut voice, 1.0, true),
            TriggerState::Low,
            "rise offset {offset}"
        );
    }
    assert_eq!(render_gate(&mut voice, 1.0, true), TriggerState::RisingEdge);
}

#[test]
fn clone_preserves_delay_pending_trigger_state_rng_and_audio() {
    let mut original = initialized_voice(3);
    let patch = Patch {
        engine: 23,
        ..Patch::default()
    };
    let modulations = Modulations {
        trigger_patched: true,
        ..Modulations::default()
    };

    for _ in 0..4 {
        render(&mut original, &patch, &modulations);
    }
    original.trigger();
    let mut cloned = original.clone();

    assert_eq!(original.trigger_delay(), cloned.trigger_delay());
    assert_eq!(original.last_trigger(), cloned.last_trigger());
    assert_eq!(original.rng().state(), cloned.rng().state());

    for block in 0..12 {
        let (original_out, original_aux) = render(&mut original, &patch, &modulations);
        let (cloned_out, cloned_aux) = render(&mut cloned, &patch, &modulations);

        assert_eq!(
            original.last_trigger(),
            cloned.last_trigger(),
            "block {block}"
        );
        let expected_trigger = if block == 0 {
            TriggerState::RisingEdge
        } else {
            TriggerState::Low
        };
        assert_eq!(original.last_trigger(), expected_trigger, "block {block}");
        assert_eq!(original_out, cloned_out, "out block {block}");
        assert_eq!(original_aux, cloned_aux, "aux block {block}");
    }
    assert_eq!(original.rng().state(), cloned.rng().state());
}

#[test]
fn explicit_trigger_produces_a_representative_audible_hit() {
    let mut control = initialized_voice(LEGACY_TRIGGER_DELAY);
    let patch = Patch {
        engine: 21,
        ..Patch::default()
    };
    let modulations = Modulations {
        trigger_patched: true,
        ..Modulations::default()
    };

    for _ in 0..32 {
        render(&mut control, &patch, &modulations);
    }
    let mut hit = control.clone();
    hit.trigger();

    let mut samples_differ = false;
    let mut control_energy = 0.0_f64;
    let mut hit_energy = 0.0_f64;
    for block in 0..64 {
        let (control_out, control_aux) = render(&mut control, &patch, &modulations);
        let (hit_out, hit_aux) = render(&mut hit, &patch, &modulations);

        if block == 0 {
            assert_eq!(control.last_trigger(), TriggerState::Low);
            assert_eq!(hit.last_trigger(), TriggerState::RisingEdge);
        }

        for ((control_sample, hit_sample), (control_aux_sample, hit_aux_sample)) in control_out
            .iter()
            .zip(hit_out.iter())
            .zip(control_aux.iter().zip(hit_aux.iter()))
        {
            assert!(control_sample.is_finite());
            assert!(hit_sample.is_finite());
            assert!(control_aux_sample.is_finite());
            assert!(hit_aux_sample.is_finite());

            samples_differ |= control_sample != hit_sample || control_aux_sample != hit_aux_sample;
            control_energy += f64::from(*control_sample).powi(2);
            control_energy += f64::from(*control_aux_sample).powi(2);
            hit_energy += f64::from(*hit_sample).powi(2);
            hit_energy += f64::from(*hit_aux_sample).powi(2);
        }
    }

    assert!(samples_differ);
    assert!(hit_energy > control_energy * 10.0 + 0.01);
}

#[test]
fn immediate_explicit_policy_sequence_is_supported() {
    let mut voice = initialized_voice(IMMEDIATE_TRIGGER_DELAY);
    let mut actual = Vec::new();

    actual.push(render_gate(&mut voice, 0.0, true));

    voice.trigger();
    actual.push(render_gate(&mut voice, 1.0, true));
    actual.push(render_gate(&mut voice, 1.0, true));
    actual.push(render_gate(&mut voice, 1.0, true));

    voice.trigger();
    actual.push(render_gate(&mut voice, 1.0, true));
    actual.push(render_gate(&mut voice, 1.0, true));

    actual.push(render_gate(&mut voice, 0.0, true));

    assert_eq!(
        actual,
        [
            TriggerState::Low,
            TriggerState::RisingEdge,
            TriggerState::High,
            TriggerState::High,
            TriggerState::RisingEdge,
            TriggerState::High,
            TriggerState::Low,
        ]
    );
}
