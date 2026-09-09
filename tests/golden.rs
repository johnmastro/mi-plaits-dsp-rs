//! Deterministic golden-test harness for voice-level audio scenarios.

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, MutexGuard};

use blake3::Hasher;
use hound::{SampleFormat, WavSpec, WavWriter};
use mi_plaits_dsp::resources::sysex::{SYX_BANK_0, SYX_BANK_1, SYX_BANK_2};
use mi_plaits_dsp::utils::random;
use mi_plaits_dsp::voice::{Modulations, NUM_ENGINES, Patch, Voice};
use serde::{Deserialize, Serialize};

const MANIFEST_SCHEMA: u32 = 1;
const HASH_ALGORITHM: &str = "blake3";
const HASH_DOMAIN: &[u8] = b"mi-plaits-dsp-golden-channel-v1\0";
const UPSTREAM_REVISION: &str = "077947ab7b728483dd147eaddaaea29dbc07c4de";
const SAMPLE_RATE: u32 = 48_000;
const BLOCK_SIZE: usize = 24;
const ONE_SECOND_IN_BLOCKS: usize = 2_000;
const TWO_SECONDS_IN_BLOCKS: usize = 4_000;
const FOUR_SECONDS_IN_BLOCKS: usize = 8_000;
const DEFAULT_SEED: u32 = 0x21;
const ALTERNATE_SEED: u32 = 0xDEAD_BEEF;
const TARGET_FEATURE_POLICY: &str = "default (no target-cpu=native)";
const ONSET_RMS_THRESHOLD: f64 = 0.0001;
const ENGINE_SWITCH_PERIODS: [(usize, &str); 2] = [
    (1, "noise_particle_engine_switch_period_1"),
    (400, "noise_particle_engine_switch_period_400"),
];

const ENGINE_NAMES: [&str; NUM_ENGINES] = [
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

static RNG_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Check,
    Record,
    Wav,
}

impl Mode {
    fn from_value(value: Option<&OsStr>) -> Result<Self, String> {
        match value.and_then(OsStr::to_str) {
            None | Some("") | Some("check") => Ok(Self::Check),
            Some("record") => Ok(Self::Record),
            Some("wav") => Ok(Self::Wav),
            Some(value) => Err(format!(
                "unknown GOLDEN mode {value:?}; expected check, record, or wav"
            )),
        }
    }

    fn from_environment() -> Result<Self, String> {
        Self::from_value(std::env::var_os("GOLDEN").as_deref())
    }
}

fn repeat_render_required(mode: Mode, canonical_target: bool) -> bool {
    mode == Mode::Record || !canonical_target
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Manifest {
    metadata: Metadata,
    #[serde(default)]
    scenario: BTreeMap<String, RecordedScenario>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Metadata {
    schema_version: u32,
    hash_algorithm: String,
    upstream_revision: String,
    rustc_verbose: String,
    target: String,
    operating_system: String,
    architecture: String,
    recording_profile: String,
    sample_rate: u32,
    block_size: usize,
    target_feature_policy: String,
    encoded_rustflags: String,
    onset_rms_threshold: String,
    recorded_at: String,
}

impl Metadata {
    fn current() -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            schema_version: MANIFEST_SCHEMA,
            hash_algorithm: HASH_ALGORITHM.to_owned(),
            upstream_revision: UPSTREAM_REVISION.to_owned(),
            rustc_verbose: command_output("rustc", &["--version", "--verbose"])?,
            target: canonical_target().to_owned(),
            operating_system: operating_system_description(),
            architecture: std::env::consts::ARCH.to_owned(),
            recording_profile: profile_name().to_owned(),
            sample_rate: SAMPLE_RATE,
            block_size: BLOCK_SIZE,
            target_feature_policy: TARGET_FEATURE_POLICY.to_owned(),
            encoded_rustflags: env!("GOLDEN_ENCODED_RUSTFLAGS").replace('\u{1f}', " "),
            onset_rms_threshold: format!("{ONSET_RMS_THRESHOLD:.4}"),
            recorded_at: command_output("date", &["+%F"])?,
        })
    }

    fn compatibility_errors(&self, actual: &Self) -> Vec<String> {
        let mut errors = Vec::new();
        compare_field(
            &mut errors,
            "schema_version",
            self.schema_version,
            actual.schema_version,
        );
        compare_field(
            &mut errors,
            "hash_algorithm",
            &self.hash_algorithm,
            &actual.hash_algorithm,
        );
        compare_field(
            &mut errors,
            "upstream_revision",
            &self.upstream_revision,
            &actual.upstream_revision,
        );
        compare_field(
            &mut errors,
            "rustc_verbose",
            &self.rustc_verbose,
            &actual.rustc_verbose,
        );
        compare_field(&mut errors, "target", &self.target, &actual.target);
        compare_field(
            &mut errors,
            "architecture",
            &self.architecture,
            &actual.architecture,
        );
        compare_field(
            &mut errors,
            "sample_rate",
            self.sample_rate,
            actual.sample_rate,
        );
        compare_field(
            &mut errors,
            "block_size",
            self.block_size,
            actual.block_size,
        );
        compare_field(
            &mut errors,
            "target_feature_policy",
            &self.target_feature_policy,
            &actual.target_feature_policy,
        );
        compare_field(
            &mut errors,
            "encoded_rustflags",
            &self.encoded_rustflags,
            &actual.encoded_rustflags,
        );
        errors
    }

    fn preserve_nonsemantic_fields(&mut self, previous: &Self) {
        self.recorded_at.clone_from(&previous.recorded_at);
        self.recording_profile
            .clone_from(&previous.recording_profile);
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct RecordedScenario {
    config: ScenarioConfig,
    out: ChannelFingerprint,
    aux: ChannelFingerprint,
}

#[derive(Debug)]
struct ScenarioRender {
    recorded: RecordedScenario,
    samples: Option<(Vec<f32>, Vec<f32>)>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ScenarioConfig {
    engine: usize,
    engine_name: String,
    seed: u32,
    blocks: usize,
    patch: PatchConfig,
    gate: Gate,
    sweep: Sweep,
    trigger_delay: TriggerDelay,
    #[serde(default)]
    modulations: ModulationConfig,
    #[serde(default, skip_serializing_if = "ResourceConfig::is_default")]
    resources: ResourceConfig,
    #[serde(default, skip_serializing_if = "ExecutionConfig::is_single")]
    execution: ExecutionConfig,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct ModulationConfig {
    level: f32,
    level_patched: bool,
}

impl ModulationConfig {
    fn to_modulations(&self, gate: &Gate) -> Modulations {
        Modulations {
            level: self.level,
            level_patched: self.level_patched,
            trigger_patched: !matches!(gate, Gate::Unpatched),
            ..Modulations::default()
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ResourceConfig {
    #[default]
    Default,
    SixOpRandomLfo {
        patch_index: usize,
        rate: u8,
        delay: u8,
        pitch_mod_depth: u8,
        amp_mod_depth: u8,
        reset_phase: bool,
        pitch_mod_sensitivity: u8,
    },
}

impl ResourceConfig {
    fn is_default(&self) -> bool {
        matches!(self, Self::Default)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ExecutionConfig {
    #[default]
    Single,
    Coupled {
        peer_engine: usize,
        peer_engine_name: String,
        peer_patch: PatchConfig,
        peer_gate: Gate,
        peer_sweep: Sweep,
        peer_trigger_delay: TriggerDelay,
        #[serde(default)]
        peer_modulations: ModulationConfig,
        peer_resources: ResourceConfig,
        initialization_order: CouplingOrder,
        render_order: CouplingOrder,
        logical_voice: CoupledVoice,
    },
    EngineSwitch {
        alternate_engine: usize,
        alternate_engine_name: String,
        alternate_patch: PatchConfig,
        alternate_gate: Gate,
        alternate_sweep: Sweep,
        alternate_trigger_delay: TriggerDelay,
        #[serde(default)]
        alternate_modulations: ModulationConfig,
        alternate_resources: ResourceConfig,
        period_blocks: usize,
    },
    CloneContinuation {
        warmup_blocks: usize,
        continuation_blocks: usize,
        render_order: CloneRenderOrder,
        logical_continuation: CloneContinuation,
    },
}

impl ExecutionConfig {
    fn is_single(&self) -> bool {
        matches!(self, Self::Single)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CouplingOrder {
    NoiseThenParticle,
    ParticleThenNoise,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CoupledVoice {
    Noise,
    Particle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CloneRenderOrder {
    OriginalThenClone,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CloneContinuation {
    Original,
    Clone,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct PatchConfig {
    note: f32,
    harmonics: f32,
    timbre: f32,
    morph: f32,
    frequency_modulation_amount: f32,
    timbre_modulation_amount: f32,
    morph_modulation_amount: f32,
    decay: f32,
    lpg_colour: f32,
}

impl Default for PatchConfig {
    fn default() -> Self {
        Self {
            note: 48.0,
            harmonics: 0.5,
            timbre: 0.5,
            morph: 0.5,
            frequency_modulation_amount: 0.0,
            timbre_modulation_amount: 0.0,
            morph_modulation_amount: 0.0,
            decay: 0.5,
            lpg_colour: 0.5,
        }
    }
}

impl PatchConfig {
    fn to_patch(&self, engine: usize) -> Patch {
        Patch {
            note: self.note,
            harmonics: self.harmonics,
            timbre: self.timbre,
            morph: self.morph,
            frequency_modulation_amount: self.frequency_modulation_amount,
            timbre_modulation_amount: self.timbre_modulation_amount,
            morph_modulation_amount: self.morph_modulation_amount,
            engine,
            decay: self.decay,
            lpg_colour: self.lpg_colour,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Gate {
    Unpatched,
    Low,
    High,
    RiseAfter { block: usize },
    Pulse { period: usize, high: usize },
    Retrigger { first_high: usize, low: usize },
    Legato { at: usize, patch_note: f32 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Sweep {
    None,
    HarmonicsRamp,
    TimbreRamp,
    MorphRamp,
    NoteRamp { from: f32, to: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum TriggerDelay {
    LegacyEight,
    Zero,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ChannelFingerprint {
    hash: String,
    sample_count: usize,
    peak: f64,
    rms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    onset_block: Option<usize>,
}

impl ChannelFingerprint {
    fn from_samples(samples: &[f32]) -> Result<Self, String> {
        let mut accumulator = ChannelAccumulator::new(samples.len());
        for block in samples.chunks(BLOCK_SIZE) {
            accumulator.update(block)?;
        }
        accumulator.finish()
    }
}

struct ChannelAccumulator {
    hasher: Hasher,
    expected_samples: usize,
    sample_count: usize,
    peak: f64,
    square_sum: f64,
    onset_block: Option<usize>,
    blocks_seen: usize,
}

struct StereoAccumulator {
    out: ChannelAccumulator,
    aux: ChannelAccumulator,
    samples: Option<(Vec<f32>, Vec<f32>)>,
}

impl StereoAccumulator {
    fn new(expected_samples: usize, capture_samples: bool) -> Self {
        Self {
            out: ChannelAccumulator::new(expected_samples),
            aux: ChannelAccumulator::new(expected_samples),
            samples: capture_samples.then(|| {
                (
                    Vec::with_capacity(expected_samples),
                    Vec::with_capacity(expected_samples),
                )
            }),
        }
    }

    fn update(&mut self, out: &[f32], aux: &[f32]) -> Result<(), String> {
        self.out
            .update(out)
            .map_err(|error| format!("out {error}"))?;
        self.aux
            .update(aux)
            .map_err(|error| format!("aux {error}"))?;
        if let Some((captured_out, captured_aux)) = &mut self.samples {
            captured_out.extend_from_slice(out);
            captured_aux.extend_from_slice(aux);
        }
        Ok(())
    }

    fn finish(self, config: ScenarioConfig) -> Result<ScenarioRender, String> {
        Ok(ScenarioRender {
            recorded: RecordedScenario {
                config,
                out: self.out.finish().map_err(|error| format!("out {error}"))?,
                aux: self.aux.finish().map_err(|error| format!("aux {error}"))?,
            },
            samples: self.samples,
        })
    }
}

impl ChannelAccumulator {
    fn new(expected_samples: usize) -> Self {
        let mut hasher = Hasher::new();
        hasher.update(HASH_DOMAIN);
        hasher.update(&(expected_samples as u64).to_le_bytes());
        Self {
            hasher,
            expected_samples,
            sample_count: 0,
            peak: 0.0,
            square_sum: 0.0,
            onset_block: None,
            blocks_seen: 0,
        }
    }

    fn update(&mut self, block: &[f32]) -> Result<(), String> {
        let mut block_square_sum = 0.0_f64;
        for (offset, sample) in block.iter().copied().enumerate() {
            if !sample.is_finite() {
                return Err(format!(
                    "sample {} is not finite: {sample:?}",
                    self.sample_count + offset
                ));
            }
            self.hasher.update(&sample.to_le_bytes());
            let sample = f64::from(sample);
            self.peak = self.peak.max(sample.abs());
            let square = sample * sample;
            self.square_sum += square;
            block_square_sum += square;
        }

        if self.onset_block.is_none()
            && !block.is_empty()
            && (block_square_sum / block.len() as f64).sqrt() > ONSET_RMS_THRESHOLD
        {
            self.onset_block = Some(self.blocks_seen);
        }
        self.sample_count += block.len();
        self.blocks_seen += 1;
        Ok(())
    }

    fn finish(self) -> Result<ChannelFingerprint, String> {
        if self.sample_count != self.expected_samples {
            return Err(format!(
                "expected {} samples but received {}",
                self.expected_samples, self.sample_count
            ));
        }
        let rms = if self.sample_count == 0 {
            0.0
        } else {
            (self.square_sum / self.sample_count as f64).sqrt()
        };
        Ok(ChannelFingerprint {
            hash: format!("{HASH_ALGORITHM}:{}", self.hasher.finalize().to_hex()),
            sample_count: self.sample_count,
            peak: round_four(self.peak),
            rms: round_four(rms),
            onset_block: self.onset_block,
        })
    }
}

fn rng_guard() -> MutexGuard<'static, ()> {
    RNG_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn seed_scenario(seed: u32) {
    random::seed(seed);
}

fn round_four(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

fn insert_scenario<T>(
    scenarios: &mut BTreeMap<String, T>,
    name: impl Into<String>,
    scenario: T,
) -> Result<(), String> {
    let name = name.into();
    match scenarios.entry(name) {
        Entry::Vacant(entry) => {
            entry.insert(scenario);
            Ok(())
        }
        Entry::Occupied(entry) => Err(format!("duplicate golden scenario name {:?}", entry.key())),
    }
}

fn scenario_config(engine: usize, gate: Gate, sweep: Sweep) -> ScenarioConfig {
    ScenarioConfig {
        engine,
        engine_name: ENGINE_NAMES[engine].to_owned(),
        seed: DEFAULT_SEED,
        blocks: TWO_SECONDS_IN_BLOCKS,
        patch: PatchConfig::default(),
        gate,
        sweep,
        trigger_delay: TriggerDelay::LegacyEight,
        modulations: ModulationConfig::default(),
        resources: ResourceConfig::Default,
        execution: ExecutionConfig::Single,
    }
}

fn core_scenario_configs() -> Result<BTreeMap<String, ScenarioConfig>, String> {
    let mut scenarios = BTreeMap::new();
    for (engine, engine_name) in ENGINE_NAMES.iter().enumerate() {
        for (suffix, gate) in [("unpatched", Gate::Unpatched), ("high", Gate::High)] {
            insert_scenario(
                &mut scenarios,
                format!("{engine_name}_sustained_{suffix}"),
                scenario_config(engine, gate, Sweep::None),
            )?;
        }

        for (suffix, sweep) in [
            ("harmonics", Sweep::HarmonicsRamp),
            ("timbre", Sweep::TimbreRamp),
            ("morph", Sweep::MorphRamp),
        ] {
            insert_scenario(
                &mut scenarios,
                format!("{engine_name}_{suffix}_ramp"),
                scenario_config(engine, Gate::Unpatched, sweep),
            )?;
        }

        for (suffix, gate) in [
            ("rise_after_2000", Gate::RiseAfter { block: 2_000 }),
            (
                "pulse_400_200",
                Gate::Pulse {
                    period: 400,
                    high: 200,
                },
            ),
            (
                "retrigger_2000_1",
                Gate::Retrigger {
                    first_high: 2_000,
                    low: 1,
                },
            ),
            (
                "legato_1000_note_55",
                Gate::Legato {
                    at: 1_000,
                    patch_note: 55.0,
                },
            ),
        ] {
            insert_scenario(
                &mut scenarios,
                format!("{engine_name}_{suffix}"),
                scenario_config(engine, gate, Sweep::None),
            )?;
        }
    }

    let expected = NUM_ENGINES * (2 + 3 + 4);
    if scenarios.len() != expected {
        return Err(format!(
            "expected {expected} core scenarios, built {}",
            scenarios.len()
        ));
    }
    Ok(scenarios)
}

fn level_cv_trigger_configs() -> Result<BTreeMap<String, ScenarioConfig>, String> {
    let mut scenarios = BTreeMap::new();

    let level_cv = ModulationConfig {
        level: 0.7,
        level_patched: true,
    };

    let mut timbre_envelope = scenario_config(0, Gate::RiseAfter { block: 2_000 }, Sweep::None);
    timbre_envelope.patch.timbre_modulation_amount = 0.5;
    timbre_envelope.modulations = level_cv.clone();
    insert_scenario(
        &mut scenarios,
        "virtual_analog_vcf_level_cv_timbre_envelope_rise_after_2000",
        timbre_envelope,
    )?;

    let mut direct_trigger = scenario_config(
        7,
        Gate::Pulse {
            period: 400,
            high: 200,
        },
        Sweep::None,
    );
    direct_trigger.modulations = level_cv;
    insert_scenario(
        &mut scenarios,
        "chiptune_level_cv_direct_trigger_pulse_400_200",
        direct_trigger,
    )?;

    Ok(scenarios)
}

fn six_op_depth_configs() -> Result<BTreeMap<String, ScenarioConfig>, String> {
    let mut scenarios = BTreeMap::new();
    for (engine, engine_name) in ENGINE_NAMES.iter().enumerate().take(5).skip(2) {
        let mut note_ramp = scenario_config(
            engine,
            Gate::Unpatched,
            Sweep::NoteRamp {
                from: 24.0,
                to: 96.0,
            },
        );
        note_ramp.blocks = FOUR_SECONDS_IN_BLOCKS;
        insert_scenario(
            &mut scenarios,
            format!("{engine_name}_note_ramp_24_96"),
            note_ramp,
        )?;

        for (parameter, sweep) in [("timbre", Sweep::TimbreRamp), ("morph", Sweep::MorphRamp)] {
            insert_scenario(
                &mut scenarios,
                format!("{engine_name}_{parameter}_ramp_pulse_400_200"),
                scenario_config(
                    engine,
                    Gate::Pulse {
                        period: 400,
                        high: 200,
                    },
                    sweep,
                ),
            )?;
        }
    }

    if scenarios.len() != 9 {
        return Err(format!(
            "expected 9 six-op depth scenarios, built {}",
            scenarios.len()
        ));
    }
    Ok(scenarios)
}

fn random_case_configs() -> Vec<(&'static str, ScenarioConfig)> {
    let mut cases = Vec::new();

    for (engine, case_name) in [
        (2, "six_op_bank_a_random_lfo"),
        (3, "six_op_bank_b_random_lfo"),
        (4, "six_op_bank_c_random_lfo"),
    ] {
        let mut config = scenario_config(
            engine,
            Gate::Pulse {
                period: 400,
                high: 200,
            },
            Sweep::None,
        );
        config.patch.harmonics = 0.0;
        config.patch.morph = 0.8;
        config.resources = ResourceConfig::SixOpRandomLfo {
            patch_index: 0,
            rate: 99,
            delay: 0,
            pitch_mod_depth: 99,
            amp_mod_depth: 0,
            reset_phase: false,
            pitch_mod_sensitivity: 7,
        };
        cases.push((case_name, config));
    }

    let mut chiptune = scenario_config(
        7,
        Gate::Pulse {
            period: 400,
            high: 200,
        },
        Sweep::None,
    );
    chiptune.patch.timbre = 0.85;
    cases.push(("chiptune_random_arpeggiator", chiptune));

    let mut speech = scenario_config(15, Gate::Unpatched, Sweep::None);
    speech.patch.harmonics = 1.0;
    speech.patch.morph = 0.4;
    cases.push(("speech_lpc_noise", speech));

    let mut swarm = scenario_config(16, Gate::Unpatched, Sweep::None);
    swarm.patch.harmonics = 0.8;
    swarm.patch.timbre = 0.8;
    swarm.patch.morph = 0.8;
    cases.push(("swarm_grains", swarm));

    let mut noise = scenario_config(17, Gate::Unpatched, Sweep::None);
    noise.patch.timbre = 0.8;
    noise.patch.morph = 0.7;
    cases.push(("noise_clocked", noise));

    let mut particle = scenario_config(18, Gate::Unpatched, Sweep::None);
    particle.patch.harmonics = 0.8;
    particle.patch.timbre = 0.8;
    particle.patch.morph = 0.7;
    cases.push(("particle_dust", particle));

    let mut string = scenario_config(19, Gate::Unpatched, Sweep::None);
    string.patch.harmonics = 0.6;
    string.patch.timbre = 0.8;
    string.patch.morph = 0.7;
    cases.push(("string_dust", string));

    let mut modal = scenario_config(20, Gate::Unpatched, Sweep::None);
    modal.patch.harmonics = 0.6;
    modal.patch.timbre = 0.8;
    modal.patch.morph = 0.7;
    cases.push(("modal_dust", modal));

    let mut bass_drum = scenario_config(21, Gate::Unpatched, Sweep::None);
    bass_drum.patch.harmonics = 1.0;
    bass_drum.patch.timbre = 0.7;
    bass_drum.patch.morph = 0.7;
    cases.push(("bass_drum_noise", bass_drum));

    let mut snare_drum = scenario_config(22, Gate::Unpatched, Sweep::None);
    snare_drum.patch.harmonics = 0.8;
    snare_drum.patch.timbre = 0.7;
    snare_drum.patch.morph = 0.7;
    cases.push(("snare_drum_noise", snare_drum));

    let mut hihat = scenario_config(23, Gate::Unpatched, Sweep::None);
    hihat.patch.harmonics = 1.0;
    hihat.patch.timbre = 0.7;
    hihat.patch.morph = 0.7;
    cases.push(("hihat_clocked_noise", hihat));

    cases
}

fn random_scenario_configs() -> Result<BTreeMap<String, ScenarioConfig>, String> {
    let mut scenarios = BTreeMap::new();
    for (case_name, base) in random_case_configs() {
        for (seed_name, seed) in [
            ("seed_21", DEFAULT_SEED),
            ("seed_dead_beef", ALTERNATE_SEED),
        ] {
            let mut config = base.clone();
            config.seed = seed;
            insert_scenario(&mut scenarios, format!("{case_name}_{seed_name}"), config)?;
        }
    }
    Ok(scenarios)
}

fn six_op_bank_override(config: &ScenarioConfig) -> Result<Option<[u8; 4096]>, String> {
    let ResourceConfig::SixOpRandomLfo {
        patch_index,
        rate,
        delay,
        pitch_mod_depth,
        amp_mod_depth,
        reset_phase,
        pitch_mod_sensitivity,
    } = config.resources
    else {
        return Ok(None);
    };

    if !(2..=4).contains(&config.engine) {
        return Err("six-op LFO override requires engine 2, 3, or 4".to_owned());
    }
    if patch_index >= 32
        || rate > 99
        || delay > 99
        || pitch_mod_depth > 99
        || amp_mod_depth > 99
        || pitch_mod_sensitivity > 7
    {
        return Err(format!(
            "invalid six-op LFO override: patch {patch_index}, rate {rate}, delay {delay}, pitch depth {pitch_mod_depth}, amp depth {amp_mod_depth}, sensitivity {pitch_mod_sensitivity}"
        ));
    }

    let mut bank = match config.engine {
        2 => SYX_BANK_0,
        3 => SYX_BANK_1,
        4 => SYX_BANK_2,
        _ => unreachable!(),
    };
    let offset = patch_index * 128;
    bank[offset + 112] = rate;
    bank[offset + 113] = delay;
    bank[offset + 114] = pitch_mod_depth;
    bank[offset + 115] = amp_mod_depth;
    bank[offset + 116] = u8::from(reset_phase) | (5 << 1) | (pitch_mod_sensitivity << 4);
    Ok(Some(bank))
}

fn render_scenario(
    config: &ScenarioConfig,
    capture_samples: bool,
) -> Result<ScenarioRender, String> {
    if config.engine >= NUM_ENGINES {
        return Err(format!("invalid engine index {}", config.engine));
    }
    if config.engine_name != ENGINE_NAMES[config.engine] {
        return Err(format!(
            "engine {} is named {:?}, not {:?}",
            config.engine, config.engine_name, ENGINE_NAMES[config.engine]
        ));
    }
    if config.trigger_delay != TriggerDelay::LegacyEight {
        return Err("zero trigger delay is unavailable in golden scenarios".to_owned());
    }
    if !config.execution.is_single() {
        return Err("special execution config passed to single-voice renderer".to_owned());
    }

    seed_scenario(config.seed);
    let six_op_bank = six_op_bank_override(config)?;
    let mut voice = Voice::new(BLOCK_SIZE, SAMPLE_RATE as f32);
    if let Some(bank) = six_op_bank.as_ref() {
        match config.engine {
            2 => voice.resources.syx_bank_a = bank,
            3 => voice.resources.syx_bank_b = bank,
            4 => voice.resources.syx_bank_c = bank,
            _ => unreachable!(),
        }
    }
    voice.init();
    let mut patch = config.patch.to_patch(config.engine);
    let mut modulations = config.modulations.to_modulations(&config.gate);
    let expected_samples = config
        .blocks
        .checked_mul(BLOCK_SIZE)
        .ok_or_else(|| "scenario sample count overflowed usize".to_owned())?;
    let mut accumulated = StereoAccumulator::new(expected_samples, capture_samples);
    let mut out = [0.0; BLOCK_SIZE];
    let mut aux = [0.0; BLOCK_SIZE];

    for block in 0..config.blocks {
        apply_sweep(
            &mut patch,
            &config.patch,
            &config.sweep,
            block,
            config.blocks,
        );
        apply_gate(&mut patch, &mut modulations, &config.gate, block)?;
        voice.render(&patch, &modulations, &mut out, &mut aux);
        accumulated.update(&out, &aux)?;
    }

    accumulated.finish(config.clone())
}

fn apply_sweep(patch: &mut Patch, base: &PatchConfig, sweep: &Sweep, block: usize, blocks: usize) {
    patch.harmonics = base.harmonics;
    patch.timbre = base.timbre;
    patch.morph = base.morph;
    patch.note = base.note;
    let ramp = block as f32 / blocks as f32;
    match *sweep {
        Sweep::None => {}
        Sweep::HarmonicsRamp => patch.harmonics = ramp,
        Sweep::TimbreRamp => patch.timbre = ramp,
        Sweep::MorphRamp => patch.morph = ramp,
        Sweep::NoteRamp { from, to } => patch.note = from + (to - from) * ramp,
    }
}

fn apply_gate(
    patch: &mut Patch,
    modulations: &mut Modulations,
    gate: &Gate,
    block: usize,
) -> Result<(), String> {
    patch.note = match *gate {
        Gate::Legato { at, patch_note } if block >= at => patch_note,
        _ => patch.note,
    };
    modulations.trigger = match *gate {
        Gate::Unpatched | Gate::Low => 0.0,
        Gate::High | Gate::Legato { .. } => 1.0,
        Gate::RiseAfter { block: rise } => {
            if block >= rise {
                1.0
            } else {
                0.0
            }
        }
        Gate::Pulse { period, high } => {
            if period == 0 || high > period {
                return Err(format!("invalid pulse gate: period {period}, high {high}"));
            }
            if block % period < high { 1.0 } else { 0.0 }
        }
        Gate::Retrigger { first_high, low } => {
            let high_again = first_high
                .checked_add(low)
                .ok_or_else(|| "retrigger interval overflowed usize".to_owned())?;
            if block < first_high || block >= high_again {
                1.0
            } else {
                0.0
            }
        }
    };
    Ok(())
}

fn render_all_scenarios(mode: Mode) -> Result<BTreeMap<String, RecordedScenario>, Box<dyn Error>> {
    let verify_repeatability = repeat_render_required(mode, is_canonical_target());
    let mut configs = core_scenario_configs()?;
    for (name, config) in level_cv_trigger_configs()? {
        insert_scenario(&mut configs, name, config)?;
    }
    for (name, config) in random_scenario_configs()? {
        insert_scenario(&mut configs, name, config)?;
    }
    for (name, config) in six_op_depth_configs()? {
        insert_scenario(&mut configs, name, config)?;
    }
    let mut recorded = BTreeMap::new();

    for (name, config) in configs {
        let first = render_scenario(&config, mode == Mode::Wav)
            .map_err(|error| format!("scenario {name:?}: {error}"))?;
        if verify_repeatability {
            let second = render_scenario(&config, false)
                .map_err(|error| format!("scenario {name:?} repeat: {error}"))?;
            if first.recorded != second.recorded {
                return Err(format!(
                    "scenario {name:?} is not repeatable with seed {:#x}\nfirst: {:#?}\nsecond: {:#?}",
                    config.seed, first.recorded, second.recorded
                )
                .into());
            }
        }
        if let Some((out, aux)) = first.samples {
            write_scenario_wav(&name, &out, &aux)?;
        }
        insert_scenario(&mut recorded, name, first.recorded)?;
    }
    validate_random_seed_sensitivity(&recorded)?;
    validate_level_cv_gate_sensitivity(&recorded)?;
    render_and_insert_coupling_scenarios(&mut recorded, mode, verify_repeatability)?;
    render_and_insert_engine_switch(&mut recorded, mode, verify_repeatability)?;
    render_and_insert_clone_continuations(&mut recorded, mode, verify_repeatability)?;
    Ok(recorded)
}

fn validate_random_seed_sensitivity(
    scenarios: &BTreeMap<String, RecordedScenario>,
) -> Result<(), String> {
    for (case_name, _) in random_case_configs() {
        let default_name = format!("{case_name}_seed_21");
        let alternate_name = format!("{case_name}_seed_dead_beef");
        let default = scenarios
            .get(&default_name)
            .ok_or_else(|| format!("missing random scenario {default_name:?}"))?;
        let alternate = scenarios
            .get(&alternate_name)
            .ok_or_else(|| format!("missing random scenario {alternate_name:?}"))?;
        if default.out.hash == alternate.out.hash && default.aux.hash == alternate.aux.hash {
            return Err(format!(
                "random case {case_name:?} produced identical channels with seeds {DEFAULT_SEED:#x} and {ALTERNATE_SEED:#x}"
            ));
        }
    }
    Ok(())
}

fn validate_level_cv_gate_sensitivity(
    scenarios: &BTreeMap<String, RecordedScenario>,
) -> Result<(), String> {
    for (name, mut config) in level_cv_trigger_configs()? {
        let triggered = scenarios
            .get(&name)
            .ok_or_else(|| format!("missing level-CV trigger scenario {name:?}"))?;
        config.gate = Gate::Low;
        let no_edge = render_scenario(&config, false)?.recorded;
        if triggered.out.hash == no_edge.out.hash && triggered.aux.hash == no_edge.aux.hash {
            return Err(format!(
                "level-CV trigger case {name:?} produced identical PCM hashes with triggered and patched-low inputs: out={} aux={}",
                triggered.out.hash, triggered.aux.hash,
            ));
        }
    }
    Ok(())
}

fn coupling_voice_configs() -> Result<(ScenarioConfig, ScenarioConfig), String> {
    let cases = random_case_configs();
    let noise = cases
        .iter()
        .find(|(name, _)| *name == "noise_clocked")
        .map(|(_, config)| config.clone())
        .ok_or_else(|| "missing noise coupling config".to_owned())?;
    let particle = cases
        .iter()
        .find(|(name, _)| *name == "particle_dust")
        .map(|(_, config)| config.clone())
        .ok_or_else(|| "missing particle coupling config".to_owned())?;
    Ok((noise, particle))
}

fn coupled_record_config(
    logical: &ScenarioConfig,
    peer: &ScenarioConfig,
    render_order: CouplingOrder,
    logical_voice: CoupledVoice,
) -> ScenarioConfig {
    let mut config = logical.clone();
    config.execution = ExecutionConfig::Coupled {
        peer_engine: peer.engine,
        peer_engine_name: peer.engine_name.clone(),
        peer_patch: peer.patch.clone(),
        peer_gate: peer.gate.clone(),
        peer_sweep: peer.sweep.clone(),
        peer_trigger_delay: peer.trigger_delay,
        peer_modulations: peer.modulations.clone(),
        peer_resources: peer.resources,
        initialization_order: CouplingOrder::NoiseThenParticle,
        render_order,
        logical_voice,
    };
    config
}

struct CouplingRender {
    noise: ScenarioRender,
    particle: ScenarioRender,
}

fn render_coupled_voices(
    render_order: CouplingOrder,
    capture_samples: bool,
) -> Result<CouplingRender, String> {
    let (noise_config, particle_config) = coupling_voice_configs()?;
    if noise_config.blocks != particle_config.blocks || noise_config.seed != particle_config.seed {
        return Err("coupled voices must have equal block counts and a shared seed".to_owned());
    }

    seed_scenario(noise_config.seed);
    let mut noise_voice = Voice::new(BLOCK_SIZE, SAMPLE_RATE as f32);
    let mut particle_voice = Voice::new(BLOCK_SIZE, SAMPLE_RATE as f32);
    noise_voice.init();
    particle_voice.init();

    let mut noise_patch = noise_config.patch.to_patch(noise_config.engine);
    let mut particle_patch = particle_config.patch.to_patch(particle_config.engine);
    let mut noise_modulations = Modulations {
        trigger_patched: !matches!(noise_config.gate, Gate::Unpatched),
        ..Modulations::default()
    };
    let mut particle_modulations = Modulations {
        trigger_patched: !matches!(particle_config.gate, Gate::Unpatched),
        ..Modulations::default()
    };
    let expected_samples = noise_config
        .blocks
        .checked_mul(BLOCK_SIZE)
        .ok_or_else(|| "coupling sample count overflowed usize".to_owned())?;
    let mut noise_accumulated = StereoAccumulator::new(expected_samples, capture_samples);
    let mut particle_accumulated = StereoAccumulator::new(expected_samples, capture_samples);
    let mut noise_out = [0.0; BLOCK_SIZE];
    let mut noise_aux = [0.0; BLOCK_SIZE];
    let mut particle_out = [0.0; BLOCK_SIZE];
    let mut particle_aux = [0.0; BLOCK_SIZE];

    for block in 0..noise_config.blocks {
        apply_sweep(
            &mut noise_patch,
            &noise_config.patch,
            &noise_config.sweep,
            block,
            noise_config.blocks,
        );
        apply_gate(
            &mut noise_patch,
            &mut noise_modulations,
            &noise_config.gate,
            block,
        )?;
        apply_sweep(
            &mut particle_patch,
            &particle_config.patch,
            &particle_config.sweep,
            block,
            particle_config.blocks,
        );
        apply_gate(
            &mut particle_patch,
            &mut particle_modulations,
            &particle_config.gate,
            block,
        )?;

        match render_order {
            CouplingOrder::NoiseThenParticle => {
                noise_voice.render(
                    &noise_patch,
                    &noise_modulations,
                    &mut noise_out,
                    &mut noise_aux,
                );
                particle_voice.render(
                    &particle_patch,
                    &particle_modulations,
                    &mut particle_out,
                    &mut particle_aux,
                );
            }
            CouplingOrder::ParticleThenNoise => {
                particle_voice.render(
                    &particle_patch,
                    &particle_modulations,
                    &mut particle_out,
                    &mut particle_aux,
                );
                noise_voice.render(
                    &noise_patch,
                    &noise_modulations,
                    &mut noise_out,
                    &mut noise_aux,
                );
            }
        }
        noise_accumulated.update(&noise_out, &noise_aux)?;
        particle_accumulated.update(&particle_out, &particle_aux)?;
    }

    Ok(CouplingRender {
        noise: noise_accumulated.finish(coupled_record_config(
            &noise_config,
            &particle_config,
            render_order,
            CoupledVoice::Noise,
        ))?,
        particle: particle_accumulated.finish(coupled_record_config(
            &particle_config,
            &noise_config,
            render_order,
            CoupledVoice::Particle,
        ))?,
    })
}

fn same_audio(first: &RecordedScenario, second: &RecordedScenario) -> bool {
    first.out == second.out && first.aux == second.aux
}

fn validate_coupling_order_behavior(
    noise_first: &CouplingRender,
    particle_first: &CouplingRender,
) -> Result<(), String> {
    let noise_order_dependent =
        !same_audio(&noise_first.noise.recorded, &particle_first.noise.recorded);
    let particle_order_dependent = !same_audio(
        &noise_first.particle.recorded,
        &particle_first.particle.recorded,
    );
    if !noise_order_dependent && !particle_order_dependent {
        return Err("global-RNG coupling scenarios are unexpectedly order-independent".into());
    }
    Ok(())
}

fn render_and_insert_coupling_scenarios(
    scenarios: &mut BTreeMap<String, RecordedScenario>,
    mode: Mode,
    verify_repeatability: bool,
) -> Result<(), Box<dyn Error>> {
    let noise_first = render_coupled_voices(CouplingOrder::NoiseThenParticle, mode == Mode::Wav)?;
    if verify_repeatability {
        let repeat = render_coupled_voices(CouplingOrder::NoiseThenParticle, false)?;
        if !same_audio(&noise_first.noise.recorded, &repeat.noise.recorded)
            || !same_audio(&noise_first.particle.recorded, &repeat.particle.recorded)
        {
            return Err("noise-first coupling render is not repeatable".into());
        }
    }

    let particle_first =
        render_coupled_voices(CouplingOrder::ParticleThenNoise, mode == Mode::Wav)?;
    if verify_repeatability {
        let repeat = render_coupled_voices(CouplingOrder::ParticleThenNoise, false)?;
        if !same_audio(&particle_first.noise.recorded, &repeat.noise.recorded)
            || !same_audio(&particle_first.particle.recorded, &repeat.particle.recorded)
        {
            return Err("particle-first coupling render is not repeatable".into());
        }
    }

    validate_coupling_order_behavior(&noise_first, &particle_first)?;

    for (name, render) in [
        ("coupling_noise_first_noise", noise_first.noise),
        ("coupling_noise_first_particle", noise_first.particle),
        ("coupling_particle_first_noise", particle_first.noise),
        ("coupling_particle_first_particle", particle_first.particle),
    ] {
        if let Some((out, aux)) = render.samples {
            write_scenario_wav(name, &out, &aux)?;
        }
        insert_scenario(scenarios, name, render.recorded)?;
    }
    Ok(())
}

fn engine_switch_config(period_blocks: usize) -> Result<ScenarioConfig, String> {
    let (noise, particle) = coupling_voice_configs()?;
    let mut config = noise;
    config.execution = ExecutionConfig::EngineSwitch {
        alternate_engine: particle.engine,
        alternate_engine_name: particle.engine_name,
        alternate_patch: particle.patch,
        alternate_gate: particle.gate,
        alternate_sweep: particle.sweep,
        alternate_trigger_delay: particle.trigger_delay,
        alternate_modulations: particle.modulations,
        alternate_resources: particle.resources,
        period_blocks,
    };
    Ok(config)
}

fn render_engine_switch(
    period_blocks: usize,
    capture_samples: bool,
) -> Result<ScenarioRender, String> {
    let config = engine_switch_config(period_blocks)?;
    let ExecutionConfig::EngineSwitch {
        alternate_engine,
        ref alternate_engine_name,
        ref alternate_patch,
        ref alternate_gate,
        ref alternate_sweep,
        alternate_trigger_delay,
        ref alternate_modulations,
        alternate_resources,
        period_blocks,
    } = config.execution
    else {
        unreachable!();
    };
    if period_blocks == 0
        || alternate_engine >= NUM_ENGINES
        || alternate_engine_name != ENGINE_NAMES[alternate_engine]
        || alternate_trigger_delay != TriggerDelay::LegacyEight
        || !alternate_resources.is_default()
    {
        return Err("invalid engine-switch configuration".to_owned());
    }

    seed_scenario(config.seed);
    let mut voice = Voice::new(BLOCK_SIZE, SAMPLE_RATE as f32);
    voice.init();
    let mut primary_patch = config.patch.to_patch(config.engine);
    let mut secondary_patch = alternate_patch.to_patch(alternate_engine);
    let mut primary_modulations = config.modulations.to_modulations(&config.gate);
    let mut secondary_modulations = alternate_modulations.to_modulations(alternate_gate);
    let expected_samples = config
        .blocks
        .checked_mul(BLOCK_SIZE)
        .ok_or_else(|| "engine-switch sample count overflowed usize".to_owned())?;
    let mut accumulated = StereoAccumulator::new(expected_samples, capture_samples);
    let mut out = [0.0; BLOCK_SIZE];
    let mut aux = [0.0; BLOCK_SIZE];

    for block in 0..config.blocks {
        let use_primary = (block / period_blocks) % 2 == 0;
        let (patch, modulations, base, gate, sweep) = if use_primary {
            (
                &mut primary_patch,
                &mut primary_modulations,
                &config.patch,
                &config.gate,
                &config.sweep,
            )
        } else {
            (
                &mut secondary_patch,
                &mut secondary_modulations,
                alternate_patch,
                alternate_gate,
                alternate_sweep,
            )
        };
        apply_sweep(patch, base, sweep, block, config.blocks);
        apply_gate(patch, modulations, gate, block)?;
        voice.render(patch, modulations, &mut out, &mut aux);
        accumulated.update(&out, &aux)?;
    }

    accumulated.finish(config)
}

fn render_and_insert_engine_switch(
    scenarios: &mut BTreeMap<String, RecordedScenario>,
    mode: Mode,
    verify_repeatability: bool,
) -> Result<(), Box<dyn Error>> {
    for (period_blocks, name) in ENGINE_SWITCH_PERIODS {
        let first = render_engine_switch(period_blocks, mode == Mode::Wav)?;
        if verify_repeatability {
            let second = render_engine_switch(period_blocks, false)?;
            if !same_audio(&first.recorded, &second.recorded) {
                return Err(format!(
                    "engine-switch scenario with period {period_blocks} is not repeatable"
                )
                .into());
            }
        }
        if let Some((out, aux)) = first.samples {
            write_scenario_wav(name, &out, &aux)?;
        }
        insert_scenario(scenarios, name, first.recorded)?;
    }
    Ok(())
}

fn clone_case_configs() -> Vec<(&'static str, ScenarioConfig, bool)> {
    let mut cases = random_case_configs()
        .into_iter()
        .map(|(name, mut config)| {
            config.blocks = ONE_SECOND_IN_BLOCKS;
            (name, config, true)
        })
        .collect::<Vec<_>>();

    for (name, engine) in [
        ("virtual_analog_vcf_control", 0),
        ("additive_control", 12),
        ("chord_control", 14),
    ] {
        let mut config = scenario_config(engine, Gate::Unpatched, Sweep::None);
        config.blocks = ONE_SECOND_IN_BLOCKS;
        cases.push((name, config, false));
    }
    cases
}

fn clone_record_config(
    base: &ScenarioConfig,
    logical_continuation: CloneContinuation,
) -> ScenarioConfig {
    let mut config = base.clone();
    config.execution = ExecutionConfig::CloneContinuation {
        warmup_blocks: ONE_SECOND_IN_BLOCKS,
        continuation_blocks: ONE_SECOND_IN_BLOCKS,
        render_order: CloneRenderOrder::OriginalThenClone,
        logical_continuation,
    };
    config
}

struct CloneContinuationRender {
    original: ScenarioRender,
    cloned: ScenarioRender,
}

fn render_clone_continuations(
    config: &ScenarioConfig,
    capture_samples: bool,
) -> Result<CloneContinuationRender, String> {
    if config.engine >= NUM_ENGINES
        || config.engine_name != ENGINE_NAMES[config.engine]
        || config.blocks != ONE_SECOND_IN_BLOCKS
        || config.trigger_delay != TriggerDelay::LegacyEight
        || !config.execution.is_single()
    {
        return Err("invalid clone-continuation configuration".to_owned());
    }

    seed_scenario(config.seed);
    let six_op_bank = six_op_bank_override(config)?;
    let mut original_voice = Voice::new(BLOCK_SIZE, SAMPLE_RATE as f32);
    if let Some(bank) = six_op_bank.as_ref() {
        match config.engine {
            2 => original_voice.resources.syx_bank_a = bank,
            3 => original_voice.resources.syx_bank_b = bank,
            4 => original_voice.resources.syx_bank_c = bank,
            _ => unreachable!(),
        }
    }
    original_voice.init();

    let mut original_patch = config.patch.to_patch(config.engine);
    let mut original_modulations = Modulations {
        trigger_patched: !matches!(config.gate, Gate::Unpatched),
        ..Modulations::default()
    };
    let total_blocks = ONE_SECOND_IN_BLOCKS + config.blocks;
    let mut discarded_out = [0.0; BLOCK_SIZE];
    let mut discarded_aux = [0.0; BLOCK_SIZE];
    for block in 0..ONE_SECOND_IN_BLOCKS {
        apply_sweep(
            &mut original_patch,
            &config.patch,
            &config.sweep,
            block,
            total_blocks,
        );
        apply_gate(
            &mut original_patch,
            &mut original_modulations,
            &config.gate,
            block,
        )?;
        original_voice.render(
            &original_patch,
            &original_modulations,
            &mut discarded_out,
            &mut discarded_aux,
        );
    }

    let mut cloned_voice = original_voice.clone();
    let mut cloned_patch = original_patch.clone();
    let mut cloned_modulations = original_modulations.clone();
    let expected_samples = config
        .blocks
        .checked_mul(BLOCK_SIZE)
        .ok_or_else(|| "clone-continuation sample count overflowed usize".to_owned())?;
    let mut original_accumulated = StereoAccumulator::new(expected_samples, capture_samples);
    let mut cloned_accumulated = StereoAccumulator::new(expected_samples, capture_samples);
    let mut original_out = [0.0; BLOCK_SIZE];
    let mut original_aux = [0.0; BLOCK_SIZE];
    let mut cloned_out = [0.0; BLOCK_SIZE];
    let mut cloned_aux = [0.0; BLOCK_SIZE];

    for block in ONE_SECOND_IN_BLOCKS..total_blocks {
        apply_sweep(
            &mut original_patch,
            &config.patch,
            &config.sweep,
            block,
            total_blocks,
        );
        apply_gate(
            &mut original_patch,
            &mut original_modulations,
            &config.gate,
            block,
        )?;
        original_voice.render(
            &original_patch,
            &original_modulations,
            &mut original_out,
            &mut original_aux,
        );
        original_accumulated.update(&original_out, &original_aux)?;
    }

    for block in ONE_SECOND_IN_BLOCKS..total_blocks {
        apply_sweep(
            &mut cloned_patch,
            &config.patch,
            &config.sweep,
            block,
            total_blocks,
        );
        apply_gate(
            &mut cloned_patch,
            &mut cloned_modulations,
            &config.gate,
            block,
        )?;
        cloned_voice.render(
            &cloned_patch,
            &cloned_modulations,
            &mut cloned_out,
            &mut cloned_aux,
        );
        cloned_accumulated.update(&cloned_out, &cloned_aux)?;
    }

    Ok(CloneContinuationRender {
        original: original_accumulated
            .finish(clone_record_config(config, CloneContinuation::Original))?,
        cloned: cloned_accumulated.finish(clone_record_config(config, CloneContinuation::Clone))?,
    })
}

fn validate_clone_continuity_behavior(
    case_name: &str,
    output_random: bool,
    render: &CloneContinuationRender,
) -> Result<(), String> {
    let identical = same_audio(&render.original.recorded, &render.cloned.recorded);
    if !output_random && !identical {
        return Err(format!(
            "non-random clone case {case_name:?} produced different continuations"
        ));
    }
    if output_random && identical {
        return Err(format!(
            "global-RNG clone case {case_name:?} unexpectedly produced identical continuations"
        ));
    }
    Ok(())
}

fn render_and_insert_clone_continuations(
    scenarios: &mut BTreeMap<String, RecordedScenario>,
    mode: Mode,
    verify_repeatability: bool,
) -> Result<(), Box<dyn Error>> {
    for (case_name, config, output_random) in clone_case_configs() {
        let first = render_clone_continuations(&config, mode == Mode::Wav)?;
        if verify_repeatability {
            let second = render_clone_continuations(&config, false)?;
            if !same_audio(&first.original.recorded, &second.original.recorded)
                || !same_audio(&first.cloned.recorded, &second.cloned.recorded)
            {
                return Err(format!("clone case {case_name:?} is not repeatable").into());
            }
        }
        validate_clone_continuity_behavior(case_name, output_random, &first)?;

        for (role, render) in [("original", first.original), ("clone", first.cloned)] {
            let name = format!("{case_name}_{role}");
            if let Some((out, aux)) = render.samples {
                write_scenario_wav(&name, &out, &aux)?;
            }
            insert_scenario(scenarios, name, render.recorded)?;
        }
    }
    Ok(())
}

fn compare_manifests(expected: &Manifest, actual: &Manifest) -> Result<(), String> {
    let mut failures = expected.metadata.compatibility_errors(&actual.metadata);
    let names = expected
        .scenario
        .keys()
        .chain(actual.scenario.keys())
        .cloned()
        .collect::<BTreeSet<_>>();

    for name in names {
        match (expected.scenario.get(&name), actual.scenario.get(&name)) {
            (Some(expected), Some(actual)) if expected == actual => {}
            (Some(expected), Some(actual)) => {
                failures.push(format!("scenario {name:?} differs"));
                append_scenario_difference(&mut failures, expected, actual);
            }
            (Some(_), None) => failures.push(format!("scenario {name:?} is missing")),
            (None, Some(_)) => failures.push(format!("scenario {name:?} is not in the manifest")),
            (None, None) => unreachable!(),
        }
    }

    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("\n"))
    }
}

fn append_scenario_difference(
    failures: &mut Vec<String>,
    expected: &RecordedScenario,
    actual: &RecordedScenario,
) {
    if expected.config != actual.config {
        failures.push(format!("  expected config: {:#?}", expected.config));
        failures.push(format!("    actual config: {:#?}", actual.config));
    }
    append_channel_difference(failures, "out", &expected.out, &actual.out);
    append_channel_difference(failures, "aux", &expected.aux, &actual.aux);
}

fn append_channel_difference(
    failures: &mut Vec<String>,
    channel: &str,
    expected: &ChannelFingerprint,
    actual: &ChannelFingerprint,
) {
    if expected != actual {
        failures.push(format!("  {channel} expected: {expected:?}"));
        failures.push(format!("  {channel}   actual: {actual:?}"));
    }
}

fn compare_field<T>(errors: &mut Vec<String>, name: &str, expected: T, actual: T)
where
    T: PartialEq + std::fmt::Debug,
{
    if expected != actual {
        errors.push(format!(
            "incompatible manifest {name}: expected {expected:?}, actual {actual:?}"
        ));
    }
}

fn manifest_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/manifest.toml")
}

fn read_manifest(path: &Path) -> Result<Manifest, Box<dyn Error>> {
    let contents = fs::read_to_string(path)?;
    Ok(toml::from_str(&contents)?)
}

fn write_manifest_atomically(path: &Path, manifest: &Manifest) -> Result<(), Box<dyn Error>> {
    let contents = toml::to_string_pretty(manifest)?;
    let temporary = path.with_extension(format!("toml.tmp-{}", std::process::id()));
    fs::write(&temporary, contents)?;

    let parsed = read_manifest(&temporary)?;
    if parsed != *manifest {
        return Err("serialized manifest did not round-trip".into());
    }

    fs::rename(temporary, path)?;
    Ok(())
}

fn write_scenario_wav(name: &str, out: &[f32], aux: &[f32]) -> Result<(), Box<dyn Error>> {
    if out.len() != aux.len() {
        return Err(format!(
            "cannot write {name:?}: out has {} samples but aux has {}",
            out.len(),
            aux.len()
        )
        .into());
    }

    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("out/golden");
    fs::create_dir_all(&directory)?;
    let path = directory.join(format!("{name}.wav"));
    let spec = WavSpec {
        channels: 2,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for (out, aux) in out.iter().zip(aux) {
        writer.write_sample(*out)?;
        writer.write_sample(*aux)?;
    }
    writer.finalize()?;
    Ok(())
}

fn current_manifest(
    previous: Option<&Manifest>,
    scenario: BTreeMap<String, RecordedScenario>,
) -> Result<Manifest, Box<dyn Error>> {
    let mut manifest = Manifest {
        metadata: Metadata::current()?,
        scenario,
    };

    if let Some(previous) = previous
        && previous.scenario == manifest.scenario
        && metadata_only_profile_or_date_changed(&previous.metadata, &manifest.metadata)
    {
        manifest
            .metadata
            .preserve_nonsemantic_fields(&previous.metadata);
    }

    Ok(manifest)
}

fn metadata_only_profile_or_date_changed(previous: &Metadata, actual: &Metadata) -> bool {
    previous.schema_version == actual.schema_version
        && previous.hash_algorithm == actual.hash_algorithm
        && previous.upstream_revision == actual.upstream_revision
        && previous.rustc_verbose == actual.rustc_verbose
        && previous.target == actual.target
        && previous.operating_system == actual.operating_system
        && previous.architecture == actual.architecture
        && previous.sample_rate == actual.sample_rate
        && previous.block_size == actual.block_size
        && previous.target_feature_policy == actual.target_feature_policy
        && previous.encoded_rustflags == actual.encoded_rustflags
        && previous.onset_rms_threshold == actual.onset_rms_threshold
}

fn command_output(program: &str, arguments: &[&str]) -> Result<String, Box<dyn Error>> {
    let output = Command::new(program).args(arguments).output()?;
    if !output.status.success() {
        return Err(format!("{program} exited with {}", output.status).into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn operating_system_description() -> String {
    if cfg!(target_os = "macos") {
        let product = command_output("sw_vers", &["-productVersion"]);
        let build = command_output("sw_vers", &["-buildVersion"]);
        if let (Ok(product), Ok(build)) = (product, build) {
            return format!("macOS {product} ({build})");
        }
    }
    std::env::consts::OS.to_owned()
}

fn canonical_target() -> &'static str {
    if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
        "aarch64-apple-darwin"
    } else {
        "noncanonical"
    }
}

fn is_canonical_target() -> bool {
    canonical_target() == "aarch64-apple-darwin"
}

fn profile_name() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

#[test]
fn golden_all() -> Result<(), Box<dyn Error>> {
    let _guard = rng_guard();
    let mode = Mode::from_environment()?;

    if !is_canonical_target() && mode == Mode::Record {
        return Err("refusing to record goldens on a noncanonical target".into());
    }

    let path = manifest_path();
    let previous = if path.try_exists()? {
        Some(read_manifest(&path)?)
    } else {
        None
    };
    let scenarios = render_all_scenarios(mode)?;

    if !is_canonical_target() {
        eprintln!(
            "verified same-seed repeatability for {} scenarios; skipping exact golden comparison on noncanonical target {:?}",
            scenarios.len(),
            canonical_target()
        );
        return Ok(());
    }

    let actual = current_manifest(previous.as_ref(), scenarios)?;

    match mode {
        Mode::Check | Mode::Wav => {
            let expected = previous.ok_or_else(|| {
                format!(
                    "golden manifest is missing at {}; run with GOLDEN=record",
                    path.display()
                )
            })?;
            compare_manifests(&expected, &actual)?;
        }
        Mode::Record => write_manifest_atomically(&path, &actual)?,
    }

    Ok(())
}

#[test]
fn harness_primitives_are_deterministic() {
    let _guard = rng_guard();
    seed_scenario(0x21);

    assert_eq!(Mode::from_value(None), Ok(Mode::Check));
    assert_eq!(Mode::from_value(Some(OsStr::new("check"))), Ok(Mode::Check));
    assert_eq!(
        Mode::from_value(Some(OsStr::new("record"))),
        Ok(Mode::Record)
    );
    assert_eq!(Mode::from_value(Some(OsStr::new("wav"))), Ok(Mode::Wav));
    assert!(Mode::from_value(Some(OsStr::new("invalid"))).is_err());
    assert!(!repeat_render_required(Mode::Check, true));
    assert!(!repeat_render_required(Mode::Wav, true));
    assert!(repeat_render_required(Mode::Record, true));
    assert!(repeat_render_required(Mode::Check, false));
    assert!(repeat_render_required(Mode::Record, false));
    assert!(repeat_render_required(Mode::Wav, false));

    let samples = [0.0, -0.0, 1.0, -1.0, 0.25, -0.25];
    let first = ChannelFingerprint::from_samples(&samples).unwrap();
    let second = ChannelFingerprint::from_samples(&samples).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first.hash,
        "blake3:62a8a33f2c51c6ed0720991322529d6226867a7df950a2d515d64b1f007c74ef"
    );
    assert_eq!(first.sample_count, samples.len());
    assert_eq!(first.peak, 1.0);
    assert_eq!(first.rms, 0.5951);
    assert_eq!(first.onset_block, Some(0));

    let serialized = toml::to_string_pretty(&first).unwrap();
    let parsed: ChannelFingerprint = toml::from_str(&serialized).unwrap();
    assert_eq!(parsed, first);
}

#[test]
fn non_finite_samples_are_rejected() {
    for sample in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(ChannelFingerprint::from_samples(&[sample]).is_err());
    }
}

#[test]
fn scenario_manifest_round_trips_and_rejects_duplicate_names() {
    let scenario = RecordedScenario {
        config: ScenarioConfig {
            engine: 7,
            engine_name: "chiptune".to_owned(),
            seed: 0xDEAD_BEEF,
            blocks: 4_000,
            patch: PatchConfig {
                note: 48.0,
                harmonics: 0.5,
                timbre: 0.8,
                morph: 0.5,
                frequency_modulation_amount: 0.0,
                timbre_modulation_amount: 0.5,
                morph_modulation_amount: 0.5,
                decay: 0.5,
                lpg_colour: 0.5,
            },
            gate: Gate::Pulse {
                period: 400,
                high: 200,
            },
            sweep: Sweep::NoteRamp {
                from: 24.0,
                to: 96.0,
            },
            trigger_delay: TriggerDelay::LegacyEight,
            modulations: ModulationConfig::default(),
            resources: ResourceConfig::Default,
            execution: ExecutionConfig::Single,
        },
        out: ChannelFingerprint::from_samples(&[0.0; BLOCK_SIZE]).unwrap(),
        aux: ChannelFingerprint::from_samples(&[0.25; BLOCK_SIZE]).unwrap(),
    };

    let mut scenarios = BTreeMap::new();
    insert_scenario(&mut scenarios, "chiptune_random", scenario).unwrap();
    let duplicate = scenarios["chiptune_random"].clone();
    assert!(insert_scenario(&mut scenarios, "chiptune_random", duplicate).is_err());
    assert_eq!(scenarios["chiptune_random"].out.sample_count, BLOCK_SIZE);

    let manifest = Manifest {
        metadata: Metadata::current().unwrap(),
        scenario: scenarios,
    };
    let serialized = toml::to_string_pretty(&manifest).unwrap();
    let parsed: Manifest = toml::from_str(&serialized).unwrap();
    assert_eq!(parsed, manifest);
}

#[test]
fn core_scenarios_have_complete_unique_engine_coverage() {
    let scenarios = core_scenario_configs().unwrap();
    assert_eq!(scenarios.len(), NUM_ENGINES * 9);

    for (engine, engine_name) in ENGINE_NAMES.iter().enumerate() {
        let for_engine = scenarios
            .values()
            .filter(|scenario| scenario.engine == engine)
            .collect::<Vec<_>>();
        assert_eq!(for_engine.len(), 9, "engine {engine}");
        assert!(
            for_engine
                .iter()
                .all(|scenario| scenario.engine_name == *engine_name)
        );
        assert!(
            for_engine
                .iter()
                .all(|scenario| scenario.blocks == TWO_SECONDS_IN_BLOCKS)
        );
    }
}

#[test]
fn level_cv_trigger_cases_cover_rising_and_repeated_edges() {
    let _guard = rng_guard();
    let scenarios = level_cv_trigger_configs().unwrap();
    assert_eq!(scenarios.len(), 2);
    assert!(scenarios.values().all(|scenario| {
        scenario.trigger_delay == TriggerDelay::LegacyEight
            && scenario.modulations
                == (ModulationConfig {
                    level: 0.7,
                    level_patched: true,
                })
    }));
    let timbre_envelope = &scenarios["virtual_analog_vcf_level_cv_timbre_envelope_rise_after_2000"];
    assert_eq!(timbre_envelope.engine, 0);
    assert_eq!(timbre_envelope.patch.timbre_modulation_amount, 0.5);
    assert!(matches!(
        timbre_envelope.gate,
        Gate::RiseAfter { block: 2_000 }
    ));

    let direct_trigger = &scenarios["chiptune_level_cv_direct_trigger_pulse_400_200"];
    assert_eq!(direct_trigger.engine, 7);
    assert!(matches!(
        direct_trigger.gate,
        Gate::Pulse {
            period: 400,
            high: 200
        }
    ));

    let recorded = scenarios
        .iter()
        .map(|(name, config)| {
            (
                name.clone(),
                render_scenario(config, false).unwrap().recorded,
            )
        })
        .collect();
    validate_level_cv_gate_sensitivity(&recorded).unwrap();
}

#[test]
fn six_op_depth_scenarios_have_complete_coverage() {
    let scenarios = six_op_depth_configs().unwrap();
    assert_eq!(scenarios.len(), 9);

    for (engine, engine_name) in ENGINE_NAMES.iter().enumerate().take(5).skip(2) {
        let for_engine = scenarios
            .values()
            .filter(|scenario| scenario.engine == engine)
            .collect::<Vec<_>>();
        assert_eq!(for_engine.len(), 3, "engine {engine}");
        assert!(for_engine.iter().all(|scenario| {
            scenario.engine_name == *engine_name
                && scenario.trigger_delay == TriggerDelay::LegacyEight
                && scenario.resources.is_default()
                && scenario.execution.is_single()
        }));
        assert_eq!(
            for_engine
                .iter()
                .filter(|scenario| matches!(scenario.sweep, Sweep::NoteRamp { .. }))
                .count(),
            1
        );
        assert_eq!(
            for_engine
                .iter()
                .filter(|scenario| matches!(scenario.sweep, Sweep::TimbreRamp))
                .count(),
            1
        );
        assert_eq!(
            for_engine
                .iter()
                .filter(|scenario| matches!(scenario.sweep, Sweep::MorphRamp))
                .count(),
            1
        );
    }
}

#[test]
fn random_activation_cases_are_seed_sensitive() {
    let _guard = rng_guard();
    let cases = random_case_configs();
    assert_eq!(cases.len(), 13);
    assert_eq!(
        cases[..3]
            .iter()
            .map(|(case_name, _)| *case_name)
            .collect::<Vec<_>>(),
        [
            "six_op_bank_a_random_lfo",
            "six_op_bank_b_random_lfo",
            "six_op_bank_c_random_lfo",
        ]
    );
    assert_eq!(
        cases
            .iter()
            .map(|(_, config)| config.engine)
            .collect::<Vec<_>>(),
        [2, 3, 4, 7, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    );

    for (case_name, config) in cases {
        let default = render_scenario(&config, false).unwrap().recorded;
        let mut alternate_config = config;
        alternate_config.seed = ALTERNATE_SEED;
        let alternate = render_scenario(&alternate_config, false).unwrap().recorded;
        assert!(
            default.out.hash != alternate.out.hash || default.aux.hash != alternate.aux.hash,
            "random case {case_name:?} did not respond to its seed: default out={} ({}) aux={} ({}), alternate out={} ({}) aux={} ({})",
            default.out.hash,
            default.out.peak,
            default.aux.hash,
            default.aux.peak,
            alternate.out.hash,
            alternate.out.peak,
            alternate.aux.hash,
            alternate.aux.peak,
        );
    }
}

#[test]
fn coupling_order_behavior_matches_global_rng() {
    let _guard = rng_guard();
    let noise_first = render_coupled_voices(CouplingOrder::NoiseThenParticle, false).unwrap();
    let particle_first = render_coupled_voices(CouplingOrder::ParticleThenNoise, false).unwrap();
    validate_coupling_order_behavior(&noise_first, &particle_first).unwrap();
}

#[test]
fn engine_switch_cases_cover_stress_and_sustained_periods() {
    assert_eq!(
        ENGINE_SWITCH_PERIODS.map(|(period_blocks, _)| period_blocks),
        [1, 400]
    );
    for (period_blocks, name) in ENGINE_SWITCH_PERIODS {
        assert!(name.ends_with(&format!("period_{period_blocks}")));
        let config = engine_switch_config(period_blocks).unwrap();
        assert_eq!(config.engine, 17);
        assert_eq!(config.blocks, TWO_SECONDS_IN_BLOCKS);
        let ExecutionConfig::EngineSwitch {
            alternate_engine,
            period_blocks: configured_period,
            ..
        } = config.execution
        else {
            panic!("expected engine-switch execution config");
        };
        assert_eq!(alternate_engine, 18);
        assert_eq!(configured_period, period_blocks);
    }
}

#[test]
fn clone_continuity_matches_global_rng() {
    let _guard = rng_guard();
    let cases = clone_case_configs();
    assert_eq!(cases.len(), 16);
    assert_eq!(cases.iter().filter(|(_, _, random)| *random).count(), 13);

    for (case_name, config, output_random) in cases {
        let render = render_clone_continuations(&config, false).unwrap();
        validate_clone_continuity_behavior(case_name, output_random, &render).unwrap();
    }
}
