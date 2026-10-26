"""Published values, transcribed from the paper.

**These numbers are not produced by this code.**  They are the values
printed in TrustNLP 2026 pp. 600-617, kept here so that

* figures and LaTeX tables can be regenerated in the manuscript's own
  layout without hand-copying numbers;
* a run's output can be diffed against what the paper claims
  (``scripts/11_crosswalk.py``);
* the released artifact is self-describing about which numbers came from
  the paper and which came from this repository.

Anything emitted from this module is tagged ``source="paper"``.  Anything
produced by a run is tagged ``source="run"``.  The two are never merged
into one column without that tag.
"""

from __future__ import annotations

from typing import Dict, List

__all__ = [
    "TABLE3_MAIN", "TABLE4_SIGNIFICANCE", "TABLE5_IWN_ABLATION",
    "TABLE6_TRAINING_DYNAMICS", "TABLE7_COMPONENTS", "TABLE8_GROUP_COUNT",
    "TABLE9_VOCAB_CONTROL", "TABLE10_SURFACE", "TABLE11_LEAVE_ONE_OUT",
    "TABLE12_EFFICIENCY", "TABLE14_PER_SEED", "TABLE15_RHO", "TABLE16_MULTILINGUAL",
    "HEADLINE_CLAIMS", "as_rows",
]


def _r(**kw) -> Dict:
    return dict(source="paper", **kw)


# --- Table 3: main results, macro-F1 mean +/- SD over three seeds ----------
TABLE3_MAIN: List[Dict] = [
    _r(model="T5-base", family="T5-T2T", params_m=220,
       D1=0.928, D1_sd=.005, D2=0.939, D2_sd=.007, D3=0.972, D3_sd=.004,
       D4=0.748, D4_sd=.013, avg_f1=0.897, delta=-0.007, train_s=412),
    _r(model="Baseline-DistilBERT", family="Baseline", params_m=66,
       D1=0.901, D1_sd=.006, D2=0.940, D2_sd=.008, D3=0.955, D3_sd=.006,
       D4=0.749, D4_sd=.012, avg_f1=0.886, delta=-0.018, train_s=82),
    _r(model="Baseline-BERT", family="Baseline", params_m=110,
       D1=0.918, D1_sd=.004, D2=0.934, D2_sd=.007, D3=0.963, D3_sd=.005,
       D4=0.760, D4_sd=.011, avg_f1=0.894, delta=-0.010, train_s=227),
    _r(model="Baseline-RoBERTa", family="Baseline", params_m=125,
       D1=0.929, D1_sd=.004, D2=0.947, D2_sd=.006, D3=0.978, D3_sd=.003,
       D4=0.762, D4_sd=.010, avg_f1=0.904, delta=0.0, train_s=233),
    _r(model="SURGeLLM-S-DistilBERT", family="SURGeLLM-S", params_m=66,
       D1=0.911, D1_sd=.007, D2=0.961, D2_sd=.006, D3=0.925, D3_sd=.009,
       D4=0.681, D4_sd=.013, avg_f1=0.870, delta=-0.034, train_s=119),
    _r(model="SURGeLLM-S-BERT", family="SURGeLLM-S", params_m=110,
       D1=0.926, D1_sd=.005, D2=0.939, D2_sd=.007, D3=0.965, D3_sd=.004,
       D4=0.748, D4_sd=.011, avg_f1=0.894, delta=-0.010, train_s=317),
    _r(model="SURGeLLM-G-RoBERTa", family="SURGeLLM-G", params_m=125,
       D1=0.937, D1_sd=.004, D2=0.949, D2_sd=.005, D3=0.977, D3_sd=.003,
       D4=0.760, D4_sd=.010, avg_f1=0.906, delta=+0.002, train_s=327),
    _r(model="SURGeLLM-FULL-RoBERTa", family="SURGeLLM-FULL", params_m=125,
       D1=0.932, D1_sd=.005, D2=0.950, D2_sd=.006, D3=0.961, D3_sd=.005,
       D4=0.711, D4_sd=.012, avg_f1=0.889, delta=-0.015, train_s=326),
    _r(model="SURGeLLM-FULL-ALBERT", family="SURGeLLM-FULL", params_m=11,
       D1=0.918, D1_sd=.006, D2=0.961, D2_sd=.005, D3=0.957, D3_sd=.005,
       D4=0.708, D4_sd=.013, avg_f1=0.886, delta=-0.018, train_s=317),
    _r(model="SURGeLLM-IWN-RoBERTa", family="IWN", params_m=125,
       D1=0.933, D1_sd=.004, D2=0.954, D2_sd=.005, D3=0.979, D3_sd=.003,
       D4=0.892, D4_sd=.009, avg_f1=0.940, delta=+0.036, train_s=332),
    _r(model="SURGeLLM-IWN-BERT", family="IWN", params_m=110,
       D1=0.927, D1_sd=.005, D2=0.946, D2_sd=.006, D3=0.968, D3_sd=.004,
       D4=0.866, D4_sd=.010, avg_f1=0.927, delta=+0.023, train_s=322),
]

# --- Table 4: BH-corrected p-values ---------------------------------------
TABLE4_SIGNIFICANCE: List[Dict] = [
    _r(comparison="SURGeLLM-S-DistilBERT vs. Base-DistilBERT", task="D2", p_bh=0.008, significant=True),
    _r(comparison="SURGeLLM-FULL-ALBERT vs. Base-RoBERTa", task="D2", p_bh=0.011, significant=True),
    _r(comparison="SURGeLLM-IWN-RoBERTa vs. Base-RoBERTa", task="D2", p_bh=0.024, significant=True),
    _r(comparison="SURGeLLM-IWN-RoBERTa vs. Base-RoBERTa", task="D4", p_bh=0.0005, significant=True),
    _r(comparison="SURGeLLM-IWN-RoBERTa vs. SURGeLLM-FULL", task="D4", p_bh=0.0005, significant=True),
    _r(comparison="SURGeLLM-IWN-BERT vs. Base-BERT", task="D4", p_bh=0.0005, significant=True),
    _r(comparison="SURGeLLM-G-RoBERTa vs. Base-RoBERTa", task="D1", p_bh=0.063, significant=False),
    _r(comparison="SURGeLLM-S-BERT vs. Base-BERT", task="D1", p_bh=0.082, significant=False),
]

# --- Table 5: IWN ablation with D4 per-class breakdown --------------------
TABLE5_IWN_ABLATION: List[Dict] = [
    _r(variant="SURGeLLM-FULL-RoBERTa", D1=0.932, D2=0.950, D3=0.961, D4=0.711,
       hum_p=0.71, hum_r=0.79, llm_p=0.71, llm_r=0.63),
    _r(variant="SURGeLLM-IWN-RoBERTa", D1=0.933, D2=0.954, D3=0.979, D4=0.892,
       hum_p=0.89, hum_r=0.89, llm_p=0.89, llm_r=0.89),
    _r(variant="Delta (RoBERTa)", D1=+.001, D2=+.004, D3=+.018, D4=+.181,
       hum_p=+.18, hum_r=+.10, llm_p=+.18, llm_r=+.26),
    _r(variant="Delta (BERT)", D1=+.001, D2=+.006, D3=+.003, D4=+.118,
       hum_p=+.13, hum_r=+.07, llm_p=+.14, llm_r=+.18),
]

# --- Table 6: training dynamics (seed 0) ----------------------------------
TABLE6_TRAINING_DYNAMICS: List[Dict] = [
    _r(model="Baseline-DistilBERT", init_loss=0.583, final_loss=0.179, best_epoch=3, delta_loss=0.404),
    _r(model="Baseline-BERT", init_loss=0.508, final_loss=0.139, best_epoch=3, delta_loss=0.370),
    _r(model="Baseline-RoBERTa", init_loss=0.543, final_loss=0.148, best_epoch=3, delta_loss=0.395),
    _r(model="T5-base", init_loss=1.234, final_loss=0.412, best_epoch=4, delta_loss=0.822),
    _r(model="SURGeLLM-S-DistilBERT", init_loss=2.019, final_loss=0.736, best_epoch=4, delta_loss=1.282),
    _r(model="SURGeLLM-S-BERT", init_loss=1.904, final_loss=0.616, best_epoch=3, delta_loss=1.087),
    _r(model="SURGeLLM-G-RoBERTa", init_loss=1.708, final_loss=0.447, best_epoch=2, delta_loss=1.262),
    _r(model="SURGeLLM-FULL-RoBERTa", init_loss=2.086, final_loss=0.682, best_epoch=2, delta_loss=1.404),
    _r(model="SURGeLLM-FULL-ALBERT", init_loss=1.905, final_loss=0.510, best_epoch=4, delta_loss=1.395),
    _r(model="SURGeLLM-IWN-RoBERTa", init_loss=1.812, final_loss=0.421, best_epoch=3, delta_loss=1.391),
    _r(model="SURGeLLM-IWN-BERT", init_loss=1.847, final_loss=0.503, best_epoch=3, delta_loss=1.344),
]

# --- Table 7: component ablation ------------------------------------------
TABLE7_COMPONENTS: List[Dict] = [
    _r(model="Baseline-RoBERTa", backbone="RoBERTa", P=0, G=0, E=0, I=0,
       D1=0.929, D2=0.947, D3=0.978, D4=0.762, avg=0.904, delta=None),
    _r(model="SURGeLLM-G-RoBERTa", backbone="RoBERTa", P=1, G=0, E=0, I=0,
       D1=0.937, D2=0.949, D3=0.977, D4=0.760, avg=0.906, delta=+.002),
    _r(model="SURGeLLM-FULL-RoBERTa", backbone="RoBERTa", P=1, G=1, E=1, I=0,
       D1=0.932, D2=0.950, D3=0.961, D4=0.711, avg=0.889, delta=-.015),
    _r(model="SURGeLLM-IWN-RoBERTa", backbone="RoBERTa", P=1, G=1, E=1, I=1,
       D1=0.933, D2=0.954, D3=0.979, D4=0.892, avg=0.940, delta=+.036),
    _r(model="Baseline-BERT", backbone="BERT", P=0, G=0, E=0, I=0,
       D1=0.918, D2=0.934, D3=0.963, D4=0.760, avg=0.894, delta=None),
    _r(model="SURGeLLM-S-BERT", backbone="BERT", P=1, G=1, E=0, I=0,
       D1=0.926, D2=0.939, D3=0.965, D4=0.748, avg=0.894, delta=0.000),
    _r(model="SURGeLLM-IWN-BERT", backbone="BERT", P=1, G=1, E=1, I=1,
       D1=0.927, D2=0.946, D3=0.968, D4=0.866, avg=0.927, delta=+.033),
    _r(model="Baseline-DistilBERT", backbone="DistilBERT", P=0, G=0, E=0, I=0,
       D1=0.901, D2=0.940, D3=0.955, D4=0.749, avg=0.886, delta=None),
    _r(model="SURGeLLM-S-DistilBERT", backbone="DistilBERT", P=1, G=1, E=0, I=0,
       D1=0.911, D2=0.961, D3=0.925, D4=0.681, avg=0.870, delta=-.016),
    _r(model="SURGeLLM-FULL-ALBERT", backbone="ALBERT", P=1, G=1, E=1, I=0,
       D1=0.918, D2=0.961, D3=0.957, D4=0.708, avg=0.886, delta=None),
]

# --- Table 8: sensitivity to number of surgical groups --------------------
TABLE8_GROUP_COUNT: List[Dict] = [
    _r(n_groups=0,  D1=0.929, D2=0.947, D3=0.978, D4=0.762, avg=0.904, note="none, baseline"),
    _r(n_groups=5,  D1=0.931, D2=0.948, D3=0.977, D4=0.760, avg=0.904, note=""),
    _r(n_groups=10, D1=0.937, D2=0.949, D3=0.977, D4=0.760, avg=0.906, note="ours"),
    _r(n_groups=15, D1=0.935, D2=0.950, D3=0.976, D4=0.755, avg=0.904, note=""),
    _r(n_groups=20, D1=0.933, D2=0.949, D3=0.974, D4=0.748, avg=0.901, note=""),
]

# --- Table 9: random-vocabulary control -----------------------------------
TABLE9_VOCAB_CONTROL: List[Dict] = [
    _r(vocab="None (Baseline)",  D1=0.929, D2=0.947, D3=0.978, D4=0.762, avg=0.904),
    _r(vocab="Random",           D1=0.910, D2=0.928, D3=0.946, D4=0.728, avg=0.878),
    _r(vocab="Auto-extracted",   D1=0.934, D2=0.948, D3=0.974, D4=0.755, avg=0.903),
    _r(vocab="Curated",          D1=0.937, D2=0.949, D3=0.977, D4=0.760, avg=0.906),
    _r(vocab="Delta Random",     D1=-.027, D2=-.021, D3=-.031, D4=-.032, avg=-.028),
    _r(vocab="Delta Auto",       D1=-.003, D2=-.001, D3=-.003, D4=-.005, avg=-.003),
]

# --- Table 10: surface-features ablation ----------------------------------
TABLE10_SURFACE: List[Dict] = [
    _r(config="G + S (full)", D1=0.937, D2=0.949, D3=0.977, D4=0.760, avg=0.906),
    _r(config="G only",       D1=0.935, D2=0.946, D3=0.974, D4=0.749, avg=0.901),
    _r(config="S only",       D1=0.928, D2=0.945, D3=0.974, D4=0.755, avg=0.901),
    _r(config="Delta no-S",   D1=-.002, D2=-.003, D3=-.003, D4=-.011, avg=-.005),
    _r(config="Delta no-G",   D1=-.009, D2=-.004, D3=-.003, D4=-.005, avg=-.005),
]

# --- Table 11: leave-one-out per-group F1 drop ----------------------------
TABLE11_LEAVE_ONE_OUT: List[Dict] = [
    _r(group="sst_pos",     D1=-.014, D2=-.000, D3=-.001, D4=-.001),
    _r(group="sst_neg",     D1=-.011, D2=-.001, D3=-.001, D4=-.001),
    _r(group="llm_stat",    D1=-.001, D2=-.002, D3=-.005, D4=-.018),
    _r(group="llm_formal",  D1=-.001, D2=-.001, D3=-.004, D4=-.012),
    _r(group="llm_list",    D1=-.001, D2=-.001, D3=-.003, D4=-.008),
    _r(group="human_pers",  D1=-.001, D2=-.001, D3=-.003, D4=-.014),
    _r(group="human_hedge", D1=-.001, D2=-.000, D3=-.002, D4=-.006),
    _r(group="human_emo",   D1=-.002, D2=-.000, D3=-.002, D4=-.010),
    _r(group="retrieval",   D1=-.000, D2=-.011, D3=-.001, D4=-.001),
    _r(group="prompt_cot",  D1=-.000, D2=-.001, D3=-.006, D4=-.002),
]

# --- Table 12: speed-accuracy trade-off -----------------------------------
TABLE12_EFFICIENCY: List[Dict] = [
    _r(model="Baseline-DistilBERT", params_m=66, train_s=82, overhead=1.0, avg_f1=0.886, f1_per_min=0.648, eff=487.0, pareto=True),
    _r(model="SURGeLLM-S-DistilBERT", params_m=66, train_s=119, overhead=1.5, avg_f1=0.870, f1_per_min=0.439, eff=478.2, pareto=False),
    _r(model="Baseline-BERT", params_m=110, train_s=227, overhead=2.8, avg_f1=0.894, f1_per_min=0.236, eff=437.7, pareto=True),
    _r(model="Baseline-RoBERTa", params_m=125, train_s=233, overhead=2.8, avg_f1=0.904, f1_per_min=0.233, eff=431.1, pareto=True),
    _r(model="SURGeLLM-S-BERT", params_m=110, train_s=317, overhead=3.9, avg_f1=0.894, f1_per_min=0.169, eff=437.7, pareto=False),
    _r(model="SURGeLLM-FULL-ALBERT", params_m=11, train_s=317, overhead=3.9, avg_f1=0.886, f1_per_min=0.168, eff=848.0, pareto=False),
    _r(model="SURGeLLM-FULL-RoBERTa", params_m=125, train_s=326, overhead=4.0, avg_f1=0.889, f1_per_min=0.164, eff=423.9, pareto=False),
    _r(model="SURGeLLM-G-RoBERTa", params_m=125, train_s=327, overhead=4.0, avg_f1=0.906, f1_per_min=0.166, eff=432.0, pareto=True),
    _r(model="SURGeLLM-IWN-BERT", params_m=110, train_s=322, overhead=3.9, avg_f1=0.927, f1_per_min=0.173, eff=453.9, pareto=False),
    _r(model="SURGeLLM-IWN-RoBERTa", params_m=125, train_s=332, overhead=4.0, avg_f1=0.940, f1_per_min=0.170, eff=448.3, pareto=True),
    _r(model="T5-base", params_m=220, train_s=412, overhead=5.0, avg_f1=0.897, f1_per_min=0.131, eff=380.4, pareto=False),
]

# --- Table 14: per-seed Avg F1 --------------------------------------------
TABLE14_PER_SEED: List[Dict] = [
    _r(model="Baseline-RoBERTa", seed0=0.906, seed1=0.901, seed2=0.905, mean=0.904),
    _r(model="SURGeLLM-G-RoBERTa", seed0=0.908, seed1=0.902, seed2=0.908, mean=0.906),
    _r(model="SURGeLLM-FULL-RoBERTa", seed0=0.892, seed1=0.886, seed2=0.889, mean=0.889),
    _r(model="SURGeLLM-IWN-RoBERTa", seed0=0.943, seed1=0.937, seed2=0.940, mean=0.940),
    _r(model="SURGeLLM-IWN-BERT", seed0=0.929, seed1=0.924, seed2=0.928, mean=0.927),
    _r(model="T5-base", seed0=0.900, seed1=0.893, seed2=0.898, mean=0.897),
]

# --- Table 15: empirical rho_k --------------------------------------------
TABLE15_RHO: List[Dict] = [
    _r(task="D1", name="SST-2",   rho_no_iwn=1.42, rho_iwn=1.39),
    _r(task="D2", name="HotPot",  rho_no_iwn=3.71, rho_iwn=3.68),
    _r(task="D3", name="LLM-7",   rho_no_iwn=1.83, rho_iwn=1.85),
    _r(task="D4", name="HumLLM",  rho_no_iwn=0.61, rho_iwn=2.13),
]

# --- Table 16: preliminary multilingual -----------------------------------
TABLE16_MULTILINGUAL: List[Dict] = [
    _r(config="XLM-R-base baseline", french=0.917, german=0.872),
    _r(config="SURGeLLM-G-XLM-R-base (auto)", french=0.926, german=0.881),
    _r(config="Delta", french=+.009, german=+.009),
]


#: The claims a reproduction is judged against, each paired with the
#: artifact that supports or refutes it.
HEADLINE_CLAIMS: List[Dict] = [
    dict(id="C1", claim="SURGeLLM-IWN-RoBERTa reaches aggregate macro-F1 0.940",
         paper_value=0.940, artifact="results/tables/table3_main_results.csv",
         kind="absolute"),
    dict(id="C2", claim="IWN beats the strongest non-IWN variant by +0.036 avg F1",
         paper_value=0.036, artifact="results/tables/table3_main_results.csv",
         kind="delta"),
    dict(id="C3", claim="IWN improves authorship detection (D4) by +0.130 over baseline",
         paper_value=0.130, artifact="results/tables/table5_iwn_ablation.csv",
         kind="delta"),
    dict(id="C4", claim="Gate without IWN regresses on D4 (-0.051 vs baseline)",
         paper_value=-0.051, artifact="results/tables/table7_component_ablation.csv",
         kind="delta"),
    dict(id="C5", claim="Random vocabulary drops avg F1 by 0.028 vs curated",
         paper_value=-0.028, artifact="results/tables/table9_vocab_control.csv",
         kind="delta"),
    dict(id="C6", claim="Auto-extracted vocabulary recovers 99.5% of curated performance",
         paper_value=0.995, artifact="results/tables/table9_vocab_control.csv",
         kind="ratio"),
    dict(id="C7", claim="Removing surface features costs -0.011 F1 on D4",
         paper_value=-0.011, artifact="results/tables/table10_surface_ablation.csv",
         kind="delta"),
    dict(id="C8", claim="rho_4 rises from ~0.61 to ~2.13 after IWN",
         paper_value=2.13, artifact="results/tables/table15_rho_estimates.csv",
         kind="absolute"),
    dict(id="C9", claim="Shared parameters exceed 99% of the total",
         paper_value=0.99, artifact="results/tables/parameter_report.csv",
         kind="fraction"),
    dict(id="C10", claim="Gate degenerates to identity when rho_k = 0 (Proposition 2)",
         paper_value=1.0, artifact="results/tables/proposition2_degeneracy.csv",
         kind="boolean"),
    dict(id="C11", claim="Surgical overhead is ~1.2M parameters, about 1% of RoBERTa-base",
         paper_value=0.01, artifact="results/tables/parameter_report.csv",
         kind="fraction"),
    dict(id="C12", claim="Each task has a clearly dominant indicator group",
         paper_value=1.0, artifact="results/tables/table11_leave_one_out.csv",
         kind="boolean"),
]


def as_rows(table: List[Dict]) -> List[Dict]:
    """Defensive copy, so callers cannot mutate the reference constants."""
    return [dict(r) for r in table]
