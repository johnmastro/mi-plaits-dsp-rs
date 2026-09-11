//! Parameter-settling diagnostic for trigger timing policy.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use hound::{SampleFormat, WavSpec, WavWriter};
use mi_plaits_dsp::engine::TriggerState;
use mi_plaits_dsp::voice::{
    IMMEDIATE_TRIGGER_DELAY, LEGACY_TRIGGER_DELAY, Modulations, Patch, Voice,
};

const SAMPLE_RATE: u32 = 48_000;
const BLOCK_SIZE: usize = 24;
const WARMUP_BLOCKS: usize = 64;
const HELD_WARMUP_BLOCKS: usize = 128;
const POST_EDGE_BLOCKS: usize = 256;
const EARLY_BLOCKS: usize = 16;
const LATE_BLOCKS: usize = 64;
const RESIDUAL_FLOOR_DBFS: f64 = -90.0;
const SEED: u32 = 0x51A7_E123;
const REPORT_DIRECTORY: &str = "out/diagnostics/trigger-settling";
const ALWAYS_LISTENING_CASES: [&str; 6] = [
    "particle_note_up",
    "bass_drum_note_up",
    "six_op_bank_a_note_up",
    "string_note_up",
    "virtual_analog_vcf_note_up",
    "fm_held_gate_note_retrigger",
];

const ENGINE_NAMES: [&str; 24] = [
    "virtual_analog_vcf",
    "phase_distortion",
    "six_op_bank_a",
    "six_op_bank_b",
    "six_op_bank_c",
    "wave_terrain",
    "string_machine",
    "chiptune",
    "virtual_analog",
    "waveshaping",
    "fm",
    "grain",
    "additive",
    "wavetable",
    "chord",
    "speech",
    "swarm",
    "noise",
    "particle",
    "string",
    "modal",
    "bass_drum",
    "snare_drum",
    "hihat",
];

const NOTE_PATHS: [&str; 24] = [
    "direct Patch::note; filter cutoff/Q/gain interpolators",
    "direct Patch::note; direct engine frequency",
    "direct Patch::note; RisingEdge loads, following High gates/latches note",
    "direct Patch::note; RisingEdge loads, following High gates/latches note",
    "direct Patch::note; RisingEdge loads, following High gates/latches note",
    "direct Patch::note; stateful terrain oscillator",
    "direct Patch::note; stateful divide-down oscillators",
    "direct Patch::note; stateful arpeggiator/oscillators",
    "direct Patch::note; stateful oscillators",
    "direct Patch::note; stateful oscillators",
    "direct Patch::note; per-sample frequency interpolators",
    "direct Patch::note; stateful grain oscillators",
    "direct Patch::note; stateful harmonic oscillators",
    "direct Patch::note; per-sample frequency interpolator",
    "direct Patch::note; stateful chord oscillators",
    "direct Patch::note; stateful speech synthesis",
    "direct Patch::note; swarm frequency/gain interpolators",
    "direct Patch::note; f0/f1 filter interpolators",
    "direct Patch::note; stateful particles/diffuser",
    "direct Patch::note; physical-string state",
    "direct Patch::note; resonator state",
    "direct Patch::note; drum model state",
    "direct Patch::note; drum model state",
    "direct Patch::note; drum model state",
];

const KNOB_PATHS: [&str; 24] = [
    "timbre: cutoff/Q/gain ParameterInterpolator",
    "none selected",
    "none selected",
    "none selected",
    "none selected",
    "morph: SimpleParameterInterpolator",
    "timbre: one-pole smoothing",
    "none selected",
    "timbre: oscillator-mix ParameterInterpolator",
    "harmonics: shape ParameterInterpolator",
    "timbre: modulation-amount ParameterInterpolator",
    "none selected",
    "timbre: harmonic-amplitude one-pole smoothing",
    "morph: pre-filter, interpolator, and one-pole smoothing",
    "timbre: one-pole smoothing",
    "none selected",
    "morph: stateful swarm amplitude smoothing",
    "harmonics: filter-mode ParameterInterpolator",
    "none selected",
    "none selected",
    "harmonics: one-pole smoothing",
    "none selected",
    "none selected",
    "none selected",
];

#[derive(Debug, Clone, Copy)]
enum Change {
    ModulationNote { old: f32, new: f32 },
    PatchNote { old: f32, new: f32 },
    Harmonics { old: f32, new: f32 },
    Timbre { old: f32, new: f32 },
    Morph { old: f32, new: f32 },
}

impl Change {
    fn description(self) -> String {
        match self {
            Self::ModulationNote { old, new } => {
                format!("Modulations::note {old:.0} to {new:.0}")
            }
            Self::PatchNote { old, new } => format!("Patch::note {old:.0} to {new:.0}"),
            Self::Harmonics { old, new } => format!("harmonics {old:.1} to {new:.1}"),
            Self::Timbre { old, new } => format!("timbre {old:.1} to {new:.1}"),
            Self::Morph { old, new } => format!("morph {old:.1} to {new:.1}"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Case {
    name: &'static str,
    engine: usize,
    change: Change,
    mechanism: &'static str,
    held_retrigger: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct Audio {
    out: Vec<f32>,
    aux: Vec<f32>,
}

impl Audio {
    fn with_capacity() -> Self {
        Self {
            out: Vec::with_capacity(POST_EDGE_BLOCKS * BLOCK_SIZE),
            aux: Vec::with_capacity(POST_EDGE_BLOCKS * BLOCK_SIZE),
        }
    }

    fn push(&mut self, out: &[f32; BLOCK_SIZE], aux: &[f32; BLOCK_SIZE]) {
        self.out.extend_from_slice(out);
        self.aux.extend_from_slice(aux);
    }

    fn residual(&self, reference: &Self) -> Self {
        assert_eq!(self.out.len(), reference.out.len());
        assert_eq!(self.aux.len(), reference.aux.len());
        Self {
            out: self
                .out
                .iter()
                .zip(reference.out.iter())
                .map(|(sample, reference)| sample - reference)
                .collect(),
            aux: self
                .aux
                .iter()
                .zip(reference.aux.iter())
                .map(|(sample, reference)| sample - reference)
                .collect(),
        }
    }

    fn interleaved_f64(&self) -> Vec<f64> {
        self.out
            .iter()
            .zip(self.aux.iter())
            .flat_map(|(out, aux)| [f64::from(*out), f64::from(*aux)])
            .collect()
    }

    fn is_finite(&self) -> bool {
        self.out
            .iter()
            .chain(self.aux.iter())
            .all(|sample| sample.is_finite())
    }
}

struct CaseAudio {
    immediate_same: Audio,
    immediate_prepared: Audio,
    legacy_gate: Audio,
}

#[derive(Debug, Clone, Copy)]
struct Metrics {
    peak_residual: f64,
    rms_residual: f64,
    peak_dbfs: f64,
    early_relative_rms_db: f64,
    late_relative_rms_db: f64,
    last_block_above_floor: Option<usize>,
}

struct CaseResult {
    case: Case,
    same_vs_prepared: Metrics,
    same_vs_legacy: Metrics,
    audio: CaseAudio,
}

fn inputs(case: Case, new_values: bool, trigger: f32) -> (Patch, Modulations) {
    let mut patch = Patch {
        note: 60.0,
        harmonics: 0.35,
        timbre: 0.35,
        morph: 0.35,
        engine: case.engine,
        ..Patch::default()
    };
    let mut modulations = Modulations {
        trigger,
        trigger_patched: true,
        ..Modulations::default()
    };

    match case.change {
        Change::ModulationNote { old, new } => {
            modulations.note = if new_values { new } else { old };
        }
        Change::PatchNote { old, new } => {
            patch.note = if new_values { new } else { old };
        }
        Change::Harmonics { old, new } => {
            patch.harmonics = if new_values { new } else { old };
        }
        Change::Timbre { old, new } => {
            patch.timbre = if new_values { new } else { old };
        }
        Change::Morph { old, new } => {
            patch.morph = if new_values { new } else { old };
        }
    }

    (patch, modulations)
}

fn new_voice(delay: usize) -> Voice<'static> {
    let mut voice = Voice::new_with_trigger_delay(BLOCK_SIZE, SAMPLE_RATE as f32, delay);
    voice.seed_rng(SEED);
    voice.init();
    voice
}

fn render_block(
    voice: &mut Voice<'static>,
    case: Case,
    new_values: bool,
    trigger: f32,
) -> ([f32; BLOCK_SIZE], [f32; BLOCK_SIZE]) {
    let (patch, modulations) = inputs(case, new_values, trigger);
    let mut out = [0.0; BLOCK_SIZE];
    let mut aux = [0.0; BLOCK_SIZE];
    voice.render(&patch, &modulations, &mut out, &mut aux);
    (out, aux)
}

fn warm_silent(case: Case, delay: usize) -> Voice<'static> {
    let mut voice = new_voice(delay);
    for _ in 0..WARMUP_BLOCKS {
        render_block(&mut voice, case, false, 0.0);
    }
    voice
}

fn warm_held(case: Case, delay: usize) -> Voice<'static> {
    let mut voice = warm_silent(case, delay);
    voice.trigger();
    render_block(&mut voice, case, false, 1.0);
    assert_eq!(voice.last_trigger(), TriggerState::RisingEdge);
    for _ in 0..HELD_WARMUP_BLOCKS {
        render_block(&mut voice, case, false, 1.0);
        assert_eq!(voice.last_trigger(), TriggerState::High);
    }
    voice
}

fn capture_following_blocks(
    voice: &mut Voice<'static>,
    case: Case,
    first: ([f32; BLOCK_SIZE], [f32; BLOCK_SIZE]),
) -> Audio {
    let mut audio = Audio::with_capacity();
    audio.push(&first.0, &first.1);
    for _ in 1..POST_EDGE_BLOCKS {
        let (out, aux) = render_block(voice, case, true, 1.0);
        audio.push(&out, &aux);
    }
    assert!(audio.is_finite());
    audio
}

fn run_case(case: Case) -> CaseAudio {
    let immediate_base = if case.held_retrigger {
        warm_held(case, IMMEDIATE_TRIGGER_DELAY)
    } else {
        warm_silent(case, IMMEDIATE_TRIGGER_DELAY)
    };
    let mut immediate_same = immediate_base.clone();
    let mut immediate_prepared = immediate_base;
    let pre_trigger_gate = if case.held_retrigger { 1.0 } else { 0.0 };

    render_block(&mut immediate_same, case, false, pre_trigger_gate);
    render_block(&mut immediate_prepared, case, true, pre_trigger_gate);

    immediate_same.trigger();
    immediate_prepared.trigger();
    let same_edge = render_block(&mut immediate_same, case, true, 1.0);
    let prepared_edge = render_block(&mut immediate_prepared, case, true, 1.0);
    assert_eq!(immediate_same.last_trigger(), TriggerState::RisingEdge);
    assert_eq!(immediate_prepared.last_trigger(), TriggerState::RisingEdge);

    let mut legacy = if case.held_retrigger {
        warm_held(case, LEGACY_TRIGGER_DELAY)
    } else {
        warm_silent(case, LEGACY_TRIGGER_DELAY)
    };
    render_block(&mut legacy, case, false, pre_trigger_gate);

    let legacy_edge = if case.held_retrigger {
        render_block(&mut legacy, case, true, 0.0);
        assert_eq!(legacy.last_trigger(), TriggerState::High);
        let mut edge = None;
        for offset in 1..=(LEGACY_TRIGGER_DELAY + 1) {
            let block = render_block(&mut legacy, case, true, 1.0);
            if offset == LEGACY_TRIGGER_DELAY {
                assert_eq!(legacy.last_trigger(), TriggerState::Low);
            }
            if offset == LEGACY_TRIGGER_DELAY + 1 {
                assert_eq!(legacy.last_trigger(), TriggerState::RisingEdge);
                edge = Some(block);
            }
        }
        edge.unwrap()
    } else {
        let mut edge = None;
        for offset in 0..=LEGACY_TRIGGER_DELAY {
            let block = render_block(&mut legacy, case, true, 1.0);
            if offset < LEGACY_TRIGGER_DELAY {
                assert_eq!(legacy.last_trigger(), TriggerState::Low);
            } else {
                assert_eq!(legacy.last_trigger(), TriggerState::RisingEdge);
                edge = Some(block);
            }
        }
        edge.unwrap()
    };

    CaseAudio {
        immediate_same: capture_following_blocks(&mut immediate_same, case, same_edge),
        immediate_prepared: capture_following_blocks(&mut immediate_prepared, case, prepared_edge),
        legacy_gate: capture_following_blocks(&mut legacy, case, legacy_edge),
    }
}

fn peak(samples: &[f64]) -> f64 {
    samples
        .iter()
        .fold(0.0_f64, |peak, sample| peak.max(sample.abs()))
}

fn rms(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|sample| sample * sample).sum::<f64>() / samples.len() as f64).sqrt()
}

fn amplitude_db(amplitude: f64) -> f64 {
    if amplitude == 0.0 {
        f64::NEG_INFINITY
    } else {
        20.0 * amplitude.log10()
    }
}

fn relative_rms_db(residual: &[f64], reference: &[f64]) -> f64 {
    let residual_rms = rms(residual);
    let reference_rms = rms(reference);
    if residual_rms == 0.0 {
        f64::NEG_INFINITY
    } else if reference_rms == 0.0 {
        f64::INFINITY
    } else {
        amplitude_db(residual_rms / reference_rms)
    }
}

fn last_block_above_floor(residual: &[f64], samples_per_block: usize, floor: f64) -> Option<usize> {
    residual
        .chunks(samples_per_block)
        .enumerate()
        .filter_map(|(block, samples)| (rms(samples) > floor).then_some(block))
        .next_back()
}

fn analyze(candidate: &Audio, reference: &Audio) -> Metrics {
    let candidate = candidate.interleaved_f64();
    let reference = reference.interleaved_f64();
    assert_eq!(candidate.len(), reference.len());
    let residual = candidate
        .iter()
        .zip(reference.iter())
        .map(|(candidate, reference)| candidate - reference)
        .collect::<Vec<_>>();
    let early_end = EARLY_BLOCKS * BLOCK_SIZE * 2;
    let late_start = (POST_EDGE_BLOCKS - LATE_BLOCKS) * BLOCK_SIZE * 2;
    let peak_residual = peak(&residual);
    let rms_residual = rms(&residual);

    Metrics {
        peak_residual,
        rms_residual,
        peak_dbfs: amplitude_db(peak_residual),
        early_relative_rms_db: relative_rms_db(&residual[..early_end], &reference[..early_end]),
        late_relative_rms_db: relative_rms_db(&residual[late_start..], &reference[late_start..]),
        last_block_above_floor: last_block_above_floor(
            &residual,
            BLOCK_SIZE * 2,
            10.0_f64.powf(RESIDUAL_FLOOR_DBFS / 20.0),
        ),
    }
}

fn result(case: Case) -> CaseResult {
    let audio = run_case(case);
    CaseResult {
        case,
        same_vs_prepared: analyze(&audio.immediate_same, &audio.immediate_prepared),
        same_vs_legacy: analyze(&audio.immediate_same, &audio.legacy_gate),
        audio,
    }
}

fn diagnostic_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for (engine, engine_name) in ENGINE_NAMES.iter().copied().enumerate() {
        cases.push(Case {
            name: Box::leak(format!("{engine_name}_note_up").into_boxed_str()),
            engine,
            change: Change::PatchNote {
                old: 36.0,
                new: 84.0,
            },
            mechanism: NOTE_PATHS[engine],
            held_retrigger: false,
        });
        cases.push(Case {
            name: Box::leak(format!("{engine_name}_note_down").into_boxed_str()),
            engine,
            change: Change::PatchNote {
                old: 84.0,
                new: 36.0,
            },
            mechanism: NOTE_PATHS[engine],
            held_retrigger: false,
        });
    }

    cases.extend([
        Case {
            name: "virtual_analog_vcf_timbre_jump",
            engine: 0,
            change: Change::Timbre { old: 0.2, new: 0.8 },
            mechanism: KNOB_PATHS[0],
            held_retrigger: false,
        },
        Case {
            name: "wave_terrain_morph_jump",
            engine: 5,
            change: Change::Morph { old: 0.2, new: 0.8 },
            mechanism: KNOB_PATHS[5],
            held_retrigger: false,
        },
        Case {
            name: "string_machine_timbre_jump",
            engine: 6,
            change: Change::Timbre { old: 0.2, new: 0.8 },
            mechanism: KNOB_PATHS[6],
            held_retrigger: false,
        },
        Case {
            name: "virtual_analog_timbre_jump",
            engine: 8,
            change: Change::Timbre { old: 0.2, new: 0.8 },
            mechanism: KNOB_PATHS[8],
            held_retrigger: false,
        },
        Case {
            name: "waveshaping_harmonics_jump",
            engine: 9,
            change: Change::Harmonics { old: 0.2, new: 0.8 },
            mechanism: KNOB_PATHS[9],
            held_retrigger: false,
        },
        Case {
            name: "fm_timbre_jump",
            engine: 10,
            change: Change::Timbre { old: 0.2, new: 0.8 },
            mechanism: KNOB_PATHS[10],
            held_retrigger: false,
        },
        Case {
            name: "additive_timbre_jump",
            engine: 12,
            change: Change::Timbre { old: 0.2, new: 0.8 },
            mechanism: KNOB_PATHS[12],
            held_retrigger: false,
        },
        Case {
            name: "wavetable_morph_jump",
            engine: 13,
            change: Change::Morph { old: 0.2, new: 0.8 },
            mechanism: KNOB_PATHS[13],
            held_retrigger: false,
        },
        Case {
            name: "chord_timbre_jump",
            engine: 14,
            change: Change::Timbre { old: 0.2, new: 0.8 },
            mechanism: KNOB_PATHS[14],
            held_retrigger: false,
        },
        Case {
            name: "swarm_morph_jump",
            engine: 16,
            change: Change::Morph { old: 0.2, new: 0.8 },
            mechanism: KNOB_PATHS[16],
            held_retrigger: false,
        },
        Case {
            name: "noise_harmonics_jump",
            engine: 17,
            change: Change::Harmonics { old: 0.2, new: 0.8 },
            mechanism: KNOB_PATHS[17],
            held_retrigger: false,
        },
        Case {
            name: "modal_harmonics_jump",
            engine: 20,
            change: Change::Harmonics { old: 0.2, new: 0.8 },
            mechanism: KNOB_PATHS[20],
            held_retrigger: false,
        },
        Case {
            name: "fm_held_gate_note_retrigger",
            engine: 10,
            change: Change::PatchNote {
                old: 36.0,
                new: 84.0,
            },
            mechanism: "held-gate explicit retrigger; prepared history changes a sounding voice",
            held_retrigger: true,
        },
    ]);
    cases
}

fn modulation_note_baseline_cases() -> Vec<Case> {
    diagnostic_cases()
        .into_iter()
        .map(|mut case| {
            if let Change::PatchNote { old, new } = case.change {
                case.change = Change::ModulationNote {
                    old: old - 60.0,
                    new: new - 60.0,
                };
                case.mechanism = "Voice modulation-note average; corresponding engine path";
            }
            case
        })
        .collect()
}

fn metric_character(metrics: Metrics) -> &'static str {
    match metrics.last_block_above_floor {
        None => "null",
        Some(block) if block < EARLY_BLOCKS => "attack-localized",
        Some(block) if block < POST_EDGE_BLOCKS - LATE_BLOCKS => "decaying",
        Some(_) => "continuing-state",
    }
}

fn differs_by_at_least_db(left: f64, right: f64, threshold: f64) -> bool {
    if left == right {
        false
    } else if left.is_finite() && right.is_finite() {
        (left - right).abs() >= threshold
    } else {
        true
    }
}

fn materially_differs(left: Metrics, right: Metrics) -> bool {
    metric_character(left) != metric_character(right)
        || differs_by_at_least_db(left.peak_dbfs, right.peak_dbfs, 6.0)
        || differs_by_at_least_db(left.early_relative_rms_db, right.early_relative_rms_db, 6.0)
        || differs_by_at_least_db(left.late_relative_rms_db, right.late_relative_rms_db, 6.0)
}

fn listening_case_names(
    results: &[CaseResult],
    modulation_note_results: &[CaseResult],
) -> Vec<&'static str> {
    assert_eq!(results.len(), modulation_note_results.len());
    results
        .iter()
        .zip(modulation_note_results)
        .filter_map(|(patch, modulation)| {
            assert_eq!(patch.case.name, modulation.case.name);
            (ALWAYS_LISTENING_CASES.contains(&patch.case.name)
                || materially_differs(patch.same_vs_prepared, modulation.same_vs_prepared))
            .then_some(patch.case.name)
        })
        .collect()
}

fn format_db(value: f64) -> String {
    if value == f64::NEG_INFINITY {
        "-inf".to_owned()
    } else if value == f64::INFINITY {
        "+inf".to_owned()
    } else {
        format!("{value:.1}")
    }
}

fn format_last_block(value: Option<usize>) -> String {
    value.map_or_else(|| "none".to_owned(), |block| block.to_string())
}

fn combined_rms(audio: &Audio) -> f64 {
    rms(&audio.interleaved_f64())
}

fn combined_peak(audio: &Audio) -> f64 {
    peak(&audio.interleaved_f64())
}

fn write_wav(path: &Path, audio: &Audio, gain: f64) {
    let spec = WavSpec {
        channels: 2,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(path, spec).unwrap();
    for (out, aux) in audio.out.iter().zip(audio.aux.iter()) {
        writer.write_sample((*out as f64 * gain) as f32).unwrap();
        writer.write_sample((*aux as f64 * gain) as f32).unwrap();
    }
    writer.finalize().unwrap();
}

fn write_listening_artifacts(directory: &Path, result: &CaseResult, filename_stem: &str) {
    let variants = [
        ("immediate_same", &result.audio.immediate_same),
        ("immediate_prepared", &result.audio.immediate_prepared),
        ("legacy_gate", &result.audio.legacy_gate),
    ];
    let target_rms = variants
        .iter()
        .map(|(_, audio)| {
            let audio_rms = combined_rms(audio);
            let audio_peak = combined_peak(audio);
            if audio_rms == 0.0 || audio_peak == 0.0 {
                0.0
            } else {
                (0.9 * audio_rms / audio_peak).min(10.0_f64.powf(-18.0 / 20.0))
            }
        })
        .fold(f64::INFINITY, f64::min);

    for (variant, audio) in variants {
        let audio_rms = combined_rms(audio);
        let gain = if audio_rms == 0.0 {
            1.0
        } else {
            target_rms / audio_rms
        };
        write_wav(
            &directory.join(format!("{filename_stem}_{variant}.wav")),
            audio,
            gain,
        );
    }

    let residual = result
        .audio
        .immediate_same
        .residual(&result.audio.immediate_prepared);
    let residual_peak = combined_peak(&residual);
    let gain = if residual_peak == 0.0 {
        1.0
    } else {
        0.8 / residual_peak
    };
    write_wav(
        &directory.join(format!("{filename_stem}_boosted_residual.wav")),
        &residual,
        gain,
    );
}

fn build_report(
    results: &[CaseResult],
    modulation_note_results: &[CaseResult],
    null_metrics: Metrics,
    positive_metrics: Metrics,
) -> String {
    let null_count = results
        .iter()
        .filter(|result| metric_character(result.same_vs_prepared) == "null")
        .count();
    let localized_count = results
        .iter()
        .filter(|result| metric_character(result.same_vs_prepared) == "attack-localized")
        .count();
    let decaying_count = results
        .iter()
        .filter(|result| metric_character(result.same_vs_prepared) == "decaying")
        .count();
    let continuing_count = results.len() - null_count - localized_count - decaying_count;
    let modulation_null_count = modulation_note_results
        .iter()
        .filter(|result| metric_character(result.same_vs_prepared) == "null")
        .count();
    let modulation_localized_count = modulation_note_results
        .iter()
        .filter(|result| metric_character(result.same_vs_prepared) == "attack-localized")
        .count();
    let modulation_decaying_count = modulation_note_results
        .iter()
        .filter(|result| metric_character(result.same_vs_prepared) == "decaying")
        .count();
    let modulation_continuing_count = modulation_note_results.len()
        - modulation_null_count
        - modulation_localized_count
        - modulation_decaying_count;
    let six_op = results
        .iter()
        .find(|result| result.case.name == "six_op_bank_a_note_up")
        .unwrap();
    let listening_cases = listening_case_names(results, modulation_note_results);
    let mut priorities = results.iter().collect::<Vec<_>>();
    priorities.sort_by(|left, right| {
        right
            .same_vs_prepared
            .peak_dbfs
            .total_cmp(&left.same_vs_prepared.peak_dbfs)
    });

    let mut report = String::new();
    writeln!(report, "# Trigger parameter-settling diagnostic").unwrap();
    writeln!(report).unwrap();
    writeln!(
        report,
        "Generated from the current fork at 48 kHz with 24-frame render blocks."
    )
    .unwrap();
    writeln!(report).unwrap();
    writeln!(report, "## Decision").unwrap();
    writeln!(report).unwrap();
    writeln!(report, "Retain the application-level `LegacyGate` policy until the selected artifacts receive controlled listening review. The measurements prioritize cases; they are not audibility verdicts. `Voice::trigger()` remains an available library operation and is not wired into the kernel by this decision.").unwrap();
    writeln!(report).unwrap();
    writeln!(report, "## Method and controls").unwrap();
    writeln!(report).unwrap();
    writeln!(report, "Regenerate this report and its WAV files with `cargo test --release --test trigger_settling -- --ignored --nocapture`. The three health controls also run during ordinary `cargo test`.").unwrap();
    writeln!(report).unwrap();
    writeln!(report, "Each case compares immediate/same-render, immediate/prepared-one-block-early, and the seven-block legacy gate policy. Windows begin at the observed `RisingEdge`, not at the input command. The early window is blocks 0–15; the late window is blocks 192–255. The last-active measurement uses a {RESIDUAL_FLOOR_DBFS:.0} dBFS per-block residual-RMS floor. Random state is per voice and deterministically seeded.").unwrap();
    writeln!(report).unwrap();
    writeln!(report, "The null control is bit exact: peak residual `{}`, RMS residual `{}`, and every residual sample is zero. The FM `Patch::note` positive control reports peak `{}` dBFS and RMS `{:.8}`; it confirms that this direct-note experiment detects the FM engine's per-sample frequency interpolation.", null_metrics.peak_residual, null_metrics.rms_residual, format_db(positive_metrics.peak_dbfs), positive_metrics.rms_residual).unwrap();
    writeln!(report).unwrap();
    writeln!(report, "The prepared reference changes engine history one block before the edge. This is silent only for a known-idle voice. The held-retrigger case deliberately shows that preparation changes an already-sounding voice. Legacy comparisons contain seven pre-edge renders (eight for the low/high retrigger gap), so persistent divergence cannot be attributed solely to one interpolation block.").unwrap();
    writeln!(report).unwrap();
    writeln!(report, "## Observed summary").unwrap();
    writeln!(report).unwrap();
    writeln!(report, "Across {} cases, the immediate same-render versus prepared comparison produced {null_count} exact nulls, {localized_count} attack-localized residuals, {decaying_count} decaying residuals, and {continuing_count} residuals reaching the late window. The 48-semitone jumps set `Patch::note` directly to 36 and 84 while leaving `Modulations::note = 0`, preserving the effective pitches of the earlier modulation-note experiment.", results.len()).unwrap();
    writeln!(report).unwrap();
    writeln!(report, "All six-op note cases are exact nulls even though the rendered signal is nonzero (bank A note-up reference RMS `{:.6}`). Source inspection explains this: `RisingEdge` loads the active six-op voice, while the following `High` render gates it and latches the note after both immediate paths have settled.", combined_rms(&six_op.audio.immediate_prepared)).unwrap();
    writeln!(report).unwrap();
    writeln!(report, "## Comparison with `Modulations::note`").unwrap();
    writeln!(report).unwrap();
    writeln!(report, "The generator also reruns the earlier `Patch::note = 60`, `Modulations::note = -24/+24` cases as an in-process baseline. That path produced {modulation_null_count} nulls, {modulation_localized_count} attack-localized residuals, {modulation_decaying_count} decaying residuals, and {modulation_continuing_count} continuing-state residuals. Direct `Patch::note` therefore adds {} exact nulls and removes {} continuing-state classifications.", null_count - modulation_null_count, modulation_continuing_count - continuing_count).unwrap();
    writeln!(report).unwrap();
    writeln!(report, "For particle, string, and modal, both note directions become bit-exact between same-render and prepared timing. Snare changes from continuing-state to attack-localized. Bass drum still has a one-block-sensitive transient, but its relative residual falls from {} to {} dB in the early window and from {} to {} dB in the late window for note-up.",
        format_db(modulation_note_results.iter().find(|result| result.case.name == "bass_drum_note_up").unwrap().same_vs_prepared.early_relative_rms_db),
        format_db(results.iter().find(|result| result.case.name == "bass_drum_note_up").unwrap().same_vs_prepared.early_relative_rms_db),
        format_db(modulation_note_results.iter().find(|result| result.case.name == "bass_drum_note_up").unwrap().same_vs_prepared.late_relative_rms_db),
        format_db(results.iter().find(|result| result.case.name == "bass_drum_note_up").unwrap().same_vs_prepared.late_relative_rms_db),
    ).unwrap();
    writeln!(report).unwrap();
    writeln!(report, "Selected same-render versus prepared comparisons:").unwrap();
    writeln!(report).unwrap();
    writeln!(report, "| Case | Modulations::note peak / early / late / character | Patch::note peak / early / late / character |").unwrap();
    writeln!(report, "| --- | --- | --- |").unwrap();
    for case_name in &listening_cases {
        let modulation = modulation_note_results
            .iter()
            .find(|result| result.case.name == *case_name)
            .unwrap()
            .same_vs_prepared;
        let patch = results
            .iter()
            .find(|result| result.case.name == *case_name)
            .unwrap()
            .same_vs_prepared;
        writeln!(
            report,
            "| {case_name} | {} / {} / {} / {} | {} / {} / {} / {} |",
            format_db(modulation.peak_dbfs),
            format_db(modulation.early_relative_rms_db),
            format_db(modulation.late_relative_rms_db),
            metric_character(modulation),
            format_db(patch.peak_dbfs),
            format_db(patch.early_relative_rms_db),
            format_db(patch.late_relative_rms_db),
            metric_character(patch),
        )
        .unwrap();
    }
    writeln!(report).unwrap();
    writeln!(
        report,
        "Highest same-versus-prepared peak residuals, for listening triage:"
    )
    .unwrap();
    writeln!(report).unwrap();
    for result in priorities.into_iter().take(8) {
        writeln!(
            report,
            "- `{}`: {} dBFS peak, {} late relative-RMS dB, {}",
            result.case.name,
            format_db(result.same_vs_prepared.peak_dbfs),
            format_db(result.same_vs_prepared.late_relative_rms_db),
            metric_character(result.same_vs_prepared),
        )
        .unwrap();
    }
    writeln!(report).unwrap();
    writeln!(report, "## Smoothing-path inventory").unwrap();
    writeln!(report).unwrap();
    writeln!(report, "These note cases use `Patch::note`, which bypasses the voice-level one-block averaging applied to `Modulations::note`. Engine-local interpolation and state still apply where listed.").unwrap();
    writeln!(report).unwrap();
    writeln!(
        report,
        "| Engine | Actual note path | Selected smoothed knob path |"
    )
    .unwrap();
    writeln!(report, "| --- | --- | --- |").unwrap();
    for engine in 0..ENGINE_NAMES.len() {
        writeln!(
            report,
            "| {} | {} | {} |",
            ENGINE_NAMES[engine], NOTE_PATHS[engine], KNOB_PATHS[engine]
        )
        .unwrap();
    }
    writeln!(report).unwrap();
    writeln!(report, "## Results").unwrap();
    writeln!(report).unwrap();
    writeln!(report, "Relative RMS values compare the residual with the named reference signal. `last` is the final block above the residual floor; `continuing-state` means it reaches the late window.").unwrap();
    writeln!(report).unwrap();
    writeln!(report, "| Case | Change and mechanism | Same vs prepared: peak dBFS / early dB / late dB / last / character | Same vs legacy: peak dBFS / early dB / late dB / last / character |").unwrap();
    writeln!(report, "| --- | --- | --- | --- |").unwrap();
    for result in results {
        let prepared = result.same_vs_prepared;
        let legacy = result.same_vs_legacy;
        writeln!(
            report,
            "| {} | {}; {} | {} / {} / {} / {} / {} | {} / {} / {} / {} / {} |",
            result.case.name,
            result.case.change.description(),
            result.case.mechanism,
            format_db(prepared.peak_dbfs),
            format_db(prepared.early_relative_rms_db),
            format_db(prepared.late_relative_rms_db),
            format_last_block(prepared.last_block_above_floor),
            metric_character(prepared),
            format_db(legacy.peak_dbfs),
            format_db(legacy.early_relative_rms_db),
            format_db(legacy.late_relative_rms_db),
            format_last_block(legacy.last_block_above_floor),
            metric_character(legacy),
        )
        .unwrap();
    }
    writeln!(report).unwrap();
    writeln!(report, "## Listening artifacts").unwrap();
    writeln!(report).unwrap();
    writeln!(report, "The original listening shortlist is retained beside this report, along with every case whose direct-note route changes residual classification or shifts peak, early-window, or late-window residual level by at least 6 dB versus the modulation-note baseline ({} cases total). Unqualified filenames contain the direct `Patch::note` run; otherwise-identical baseline files include `_modulation_note_` in the name. Within each input route, the three policy variants are RMS-matched with peak headroom. The same-vs-prepared residual is separately peak-normalized and labeled `boosted_residual`; its gain must not be interpreted as natural level. Use controlled ABX before making an audibility claim.", listening_cases.len()).unwrap();
    report
}

#[test]
fn synthetic_metric_positive_control_has_known_values() {
    let reference = [0.5, -0.5, 0.5, -0.5];
    let candidate = [1.0, -1.0, 0.5, -0.5];
    let residual = candidate
        .iter()
        .zip(reference.iter())
        .map(|(candidate, reference)| candidate - reference)
        .collect::<Vec<_>>();

    assert!((peak(&residual) - 0.5).abs() < 1.0e-12);
    assert!((rms(&residual) - (0.125_f64).sqrt()).abs() < 1.0e-12);
    assert!((relative_rms_db(&residual, &reference) + 3.010_299_956_64).abs() < 1.0e-9);
    assert_eq!(last_block_above_floor(&residual, 2, 0.1), Some(0));
}

#[test]
fn null_control_is_bit_exact_and_metrics_are_zero() {
    let case = Case {
        name: "null_control",
        engine: 10,
        change: Change::PatchNote {
            old: 60.0,
            new: 60.0,
        },
        mechanism: "identical inputs through ordinary immediate comparison machinery",
        held_retrigger: false,
    };
    let audio = run_case(case);
    let residual = audio.immediate_same.residual(&audio.immediate_prepared);
    let metrics = analyze(&audio.immediate_same, &audio.immediate_prepared);

    assert_eq!(audio.immediate_same, audio.immediate_prepared);
    assert!(residual.out.iter().all(|sample| *sample == 0.0));
    assert!(residual.aux.iter().all(|sample| *sample == 0.0));
    assert_eq!(metrics.peak_residual, 0.0);
    assert_eq!(metrics.rms_residual, 0.0);
}

#[test]
fn fm_patch_note_jump_is_a_dsp_positive_control() {
    let case = Case {
        name: "fm_patch_note_positive_control",
        engine: 10,
        change: Change::PatchNote {
            old: 24.0,
            new: 96.0,
        },
        mechanism: "FM carrier and modulator ParameterInterpolator",
        held_retrigger: false,
    };
    let audio = run_case(case);
    let metrics = analyze(&audio.immediate_same, &audio.immediate_prepared);

    assert!(audio.immediate_same.is_finite());
    assert!(audio.immediate_prepared.is_finite());
    assert!(metrics.peak_residual.is_finite());
    assert!(metrics.rms_residual.is_finite());
    assert!(metrics.peak_residual > 1.0e-6);
    assert!(metrics.rms_residual > 1.0e-7);
}

#[test]
#[ignore = "generates the full diagnostic report and listening artifacts"]
fn generate_trigger_settling_report_and_listening_artifacts() {
    let null_case = Case {
        name: "null_control",
        engine: 10,
        change: Change::PatchNote {
            old: 60.0,
            new: 60.0,
        },
        mechanism: "identical inputs",
        held_retrigger: false,
    };
    let null_audio = run_case(null_case);
    let null_metrics = analyze(&null_audio.immediate_same, &null_audio.immediate_prepared);
    assert_eq!(null_metrics.peak_residual, 0.0);
    assert_eq!(null_metrics.rms_residual, 0.0);

    let positive_case = Case {
        name: "fm_patch_note_positive_control",
        engine: 10,
        change: Change::PatchNote {
            old: 24.0,
            new: 96.0,
        },
        mechanism: "FM carrier and modulator ParameterInterpolator",
        held_retrigger: false,
    };
    let positive_audio = run_case(positive_case);
    let positive_metrics = analyze(
        &positive_audio.immediate_same,
        &positive_audio.immediate_prepared,
    );
    assert!(positive_metrics.peak_residual > 1.0e-6);

    let results = diagnostic_cases()
        .into_iter()
        .map(result)
        .collect::<Vec<_>>();
    let modulation_note_results = modulation_note_baseline_cases()
        .into_iter()
        .map(result)
        .collect::<Vec<_>>();
    assert_eq!(results.len(), 61);
    assert_eq!(modulation_note_results.len(), 61);
    let six_op = results
        .iter()
        .find(|result| result.case.name == "six_op_bank_a_note_up")
        .unwrap();
    assert!(combined_rms(&six_op.audio.immediate_prepared) > 1.0e-6);
    assert_eq!(six_op.same_vs_prepared.peak_residual, 0.0);

    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join(REPORT_DIRECTORY);
    fs::create_dir_all(&directory).unwrap();
    let listening_cases = listening_case_names(&results, &modulation_note_results);
    for result in &results {
        if listening_cases.contains(&result.case.name) {
            write_listening_artifacts(&directory, result, result.case.name);
        }
    }
    for result in &modulation_note_results {
        if listening_cases.contains(&result.case.name) {
            write_listening_artifacts(
                &directory,
                result,
                &format!("{}_modulation_note", result.case.name),
            );
        }
    }

    let report = build_report(
        &results,
        &modulation_note_results,
        null_metrics,
        positive_metrics,
    );
    fs::write(directory.join("report.md"), report).unwrap();
}
