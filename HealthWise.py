from __future__ import annotations
 
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
 
def trimf(x, a, b, c):
    x_arr = np.asarray(x, dtype=float)
    y = np.zeros_like(x_arr)
    if b != a:
        idx1 = (a < x_arr) & (x_arr <= b)
        y[idx1] = (x_arr[idx1] - a) / (b - a)
    if c != b:
        idx2 = (b < x_arr) & (x_arr < c)
        y[idx2] = (c - x_arr[idx2]) / (c - b)
    y[x_arr == b] = 1.0
    return float(y) if np.isscalar(x) else y
 
 
def trapmf(x, a, b, c, d):
    x_arr = np.asarray(x, dtype=float)
    y = np.zeros_like(x_arr)
 
    idx1 = (a < x_arr) & (x_arr <= b)
    idx2 = (b < x_arr) & (x_arr <= c)
    idx3 = (c < x_arr) & (x_arr < d)
 
    if b != a:
        y[idx1] = (x_arr[idx1] - a) / (b - a)
    y[idx2] = 1.0
    if d != c:
        y[idx3] = (d - x_arr[idx3]) / (d - c)
 
    if a == b:
        y[x_arr <= b] = np.maximum(y[x_arr <= b], 1.0)
    if c == d:
        y[x_arr >= c] = np.maximum(y[x_arr >= c], 1.0)
 
    y[x_arr <= a] = 0.0 if a != b else y[x_arr <= a]
    y[x_arr >= d] = 0.0 if c != d else y[x_arr >= d]
    y[(x_arr == b) | (x_arr == c)] = 1.0
 
    return float(y) if np.isscalar(x) else y
 
 
def hedge_indeed(mu):
    return np.asarray(mu) ** 2 if not np.isscalar(mu) else float(mu) ** 2
 
 
def hedge_somewhat(mu):
    arr = np.clip(np.asarray(mu, dtype=float), 0.0, 1.0)
    return np.sqrt(arr) if not np.isscalar(mu) else float(math.sqrt(arr))
 
 
@dataclass(frozen=True)
class Rule:
    antecedents: Tuple[Tuple[str, str], ...]
    consequent: str
    description: str
 
 
class CHDFuzzyExpertSystem:
    def __init__(self):
        self.bp_universe = np.linspace(100, 200, 2001)
        self.chol_universe = np.linspace(100, 280, 1801)
        self.hr_universe = np.linspace(50, 200, 1501)
        self.chd_universe = np.linspace(0.0, 4.0, 4001)
 
        self.input_mfs = {
            "bp": {
                "low":    lambda x: trapmf(x, 100, 100, 115, 130),
                "medium": lambda x: trimf(x, 120, 145, 170),
                "high":   lambda x: trapmf(x, 160, 175, 200, 200),
            },
            "chol": {
                "low":  lambda x: trapmf(x, 100, 100, 160, 200),
                "high": lambda x: trapmf(x, 180, 220, 280, 280),
            },
            "hr": {
                "slow":     lambda x: trapmf(x, 50, 50, 65, 80),
                "moderate": lambda x: trimf(x, 70, 85, 100),
                "fast":     lambda x: trapmf(x, 90, 120, 200, 200),
            },
        }
 
        self.output_mfs = {
            "healthy": lambda x: trapmf(x, 0.0, 0.0, 0.8, 1.5),
            "middle":  lambda x: trimf(x, 1.3, 2.0, 2.7),
            "sick":    lambda x: trapmf(x, 2.5, 3.2, 4.0, 4.0),
        }
 
        self.rules = [
            Rule((("bp", "low"),    ("chol", "low"),  ("hr", "slow")),     "healthy",
                 "IF BP is Low AND Chol is Low AND HR is Slow THEN CHD is Healthy"),
            Rule((("bp", "low"),    ("chol", "low"),  ("hr", "moderate")), "healthy",
                 "IF BP is Low AND Chol is Low AND HR is Moderate THEN CHD is Healthy"),
            Rule((("bp", "medium"), ("chol", "low"),  ("hr", "moderate")), "middle",
                 "IF BP is Medium AND Chol is Low AND HR is Moderate THEN CHD is Middle"),
            Rule((("bp", "medium"), ("chol", "high"), ("hr", "slow")),     "middle",
                 "IF BP is Medium AND Chol is High AND HR is Slow THEN CHD is Middle"),
            Rule((("bp", "high"),   ("chol", "low"),  ("hr", "moderate")), "sick",
                 "IF BP is High AND Chol is Low AND HR is Moderate THEN CHD is Sick"),
            Rule((("bp", "high"),   ("chol", "high"), ("hr", "fast")),     "sick",
                 "IF BP is High AND Chol is High AND HR is Fast THEN CHD is Sick"),
        ]
 
        self.sugeno_singletons = {
            "healthy": 0.75,
            "middle":  2.00,
            "sick":    3.25,
        }
 
        self.hedged_output_mfs = {
            "indeed_healthy": lambda x: hedge_indeed(self.output_mfs["healthy"](x)),
            "middle":         lambda x: self.output_mfs["middle"](x),
            "somewhat_sick":  lambda x: hedge_somewhat(self.output_mfs["sick"](x)),
        }
 
        self.hedged_sugeno_singletons = {
            "indeed_healthy": 0.55,
            "middle": 2.00,
            "somewhat_sick": 2.95,
        }
 
        self.hedged_rules = [
            Rule(r.antecedents,
                 "indeed_healthy" if r.consequent == "healthy"
                 else "somewhat_sick" if r.consequent == "sick"
                 else "middle",
                 r.description.replace("Healthy", "Indeed Healthy").replace("Sick", "Somewhat Sick"))
            for r in self.rules
        ]
 
    def fuzzify(self, bp, chol, hr):
        return {
            "bp":   {label: float(mf(bp))   for label, mf in self.input_mfs["bp"].items()},
            "chol": {label: float(mf(chol)) for label, mf in self.input_mfs["chol"].items()},
            "hr":   {label: float(mf(hr))   for label, mf in self.input_mfs["hr"].items()},
        }
 
    def evaluate_rules(self, fuzzified, hedged=False):
        rule_set = self.hedged_rules if hedged else self.rules
        evaluated = []
        for idx, rule in enumerate(rule_set, start=1):
            degrees = [fuzzified[var][label] for var, label in rule.antecedents]
            evaluated.append({
                "rule_no": idx,
                "rule_text": rule.description,
                "antecedent_degrees": degrees,
                "firing_strength": min(degrees),
                "consequent": rule.consequent,
            })
        return evaluated
 
    def aggregate(self, evaluated_rules, hedged=False):
        aggregated = np.zeros_like(self.chd_universe)
        output_bank = self.hedged_output_mfs if hedged else self.output_mfs
        for item in evaluated_rules:
            strength = item["firing_strength"]
            if strength <= 0:
                continue
            clipped = np.minimum(strength, output_bank[item["consequent"]](self.chd_universe))
            aggregated = np.maximum(aggregated, clipped)
        return aggregated
 
    @staticmethod
    def centroid(x, mu):
        area = np.trapezoid(mu, x)
        if area == 0:
            return float("nan")
        return float(np.trapezoid(x * mu, x) / area)
 
    def sugeno(self, evaluated_rules, hedged=False):
        singletons = self.hedged_sugeno_singletons if hedged else self.sugeno_singletons
        weights = np.array([r["firing_strength"] for r in evaluated_rules], dtype=float)
        z = np.array([singletons[r["consequent"]] for r in evaluated_rules], dtype=float)
        if weights.sum() == 0:
            return float("nan")
        return float(np.dot(weights, z) / weights.sum())
 
    def infer(self, bp, chol, hr, hedged=False):
        fuzzified = self.fuzzify(bp, chol, hr)
        evaluated = self.evaluate_rules(fuzzified, hedged=hedged)
        aggregated = self.aggregate(evaluated, hedged=hedged)
        return {
            "inputs": {"bp": bp, "chol": chol, "hr": hr},
            "fuzzified": fuzzified,
            "rules": evaluated,
            "aggregated": aggregated,
            "cog": self.centroid(self.chd_universe, aggregated),
            "sugeno": self.sugeno(evaluated, hedged=hedged),
        }
 
    @staticmethod
    def linguistic_label(value):
        if np.isnan(value):
            return "No rule fired"
        if value < 1.3:
            return "Healthy region"
        elif value < 2.5:
            return "Middle region"
        return "Sick region"
 
    def sensitivity_analysis(self, baseline, steps=120):
        ranges = {
            "bp":   (100, 200),
            "chol": (100, 280),
            "hr":   (50, 200),
        }
        rows = []
        for var, (lo, hi) in ranges.items():
            xs = np.linspace(lo, hi, steps)
            ys = []
            for x in xs:
                inp = baseline.copy()
                inp[var] = float(x)
                ys.append(self.infer(inp["bp"], inp["chol"], inp["hr"])["cog"])
 
            xs = np.array(xs)
            ys = np.array(ys)
            valid = ~np.isnan(ys)
            xs, ys = xs[valid], ys[valid]
 
            if len(xs) < 2:
                score, span = np.nan, np.nan
            else:
                score = float(np.mean(np.abs(np.diff(ys) / np.diff(xs))))
                span  = float(np.nanmax(ys) - np.nanmin(ys))
 
            rows.append({
                "input_factor": var,
                "range_tested": f"{lo} to {hi}",
                "mean_abs_slope": score,
                "output_span": span,
            })
 
        return pd.DataFrame(rows).sort_values(
            by=["output_span", "mean_abs_slope"], ascending=False
        ).reset_index(drop=True)
 
    def plot_membership_functions(self, outdir):
        os.makedirs(outdir, exist_ok=True)
 
        specs = [
            (self.bp_universe,   self.input_mfs["bp"],   "Blood Pressure",  "Blood Pressure",  "membership_bp.png"),
            (self.chol_universe, self.input_mfs["chol"], "Cholesterol",      "Cholesterol",     "membership_chol.png"),
            (self.hr_universe,   self.input_mfs["hr"],   "Heart Rate",       "Heart Rate",      "membership_hr.png"),
            (self.chd_universe,  self.output_mfs,        "CHD Output",       "CHD Level",       "membership_chd.png"),
        ]
 
        for universe, mf_dict, title, xlabel, fname in specs:
            plt.figure(figsize=(10, 6))
            for label, mf in mf_dict.items():
                plt.plot(universe, mf(universe), label=label.title())
            plt.title(f"{title} Membership Functions")
            plt.xlabel(xlabel)
            plt.ylabel("Membership Degree")
            plt.ylim(-0.02, 1.05)
            plt.legend()
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, fname), dpi=200)
            plt.close()
 
        plt.figure(figsize=(10, 6))
        plt.plot(self.chd_universe, self.output_mfs["healthy"](self.chd_universe),              label="Healthy")
        plt.plot(self.chd_universe, self.hedged_output_mfs["indeed_healthy"](self.chd_universe), label="Indeed Healthy")
        plt.plot(self.chd_universe, self.output_mfs["sick"](self.chd_universe),                  label="Sick")
        plt.plot(self.chd_universe, self.hedged_output_mfs["somewhat_sick"](self.chd_universe),  label="Somewhat Sick")
        plt.title("Hedge-Modified Output Membership Functions")
        plt.xlabel("CHD Level")
        plt.ylabel("Membership Degree")
        plt.ylim(-0.02, 1.05)
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "membership_hedges.png"), dpi=200)
        plt.close()
 
    def plot_aggregated_output(self, result, title, filename):
        plt.figure(figsize=(10, 6))
        plt.plot(self.chd_universe, result["aggregated"], label="Aggregated Output")
        plt.axvline(result["cog"], linestyle="--", label=f"COG = {result['cog']:.3f}")
        plt.title(title)
        plt.xlabel("CHD Level")
        plt.ylabel("Membership Degree")
        plt.ylim(-0.02, 1.05)
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close()
 
    def plot_surface(self, outdir, fixed_hr=95.0):
        from mpl_toolkits.mplot3d import Axes3D
 
        os.makedirs(outdir, exist_ok=True)
        bp_vals   = np.linspace(100, 200, 60)
        chol_vals = np.linspace(100, 280, 60)
        BP, CHOL  = np.meshgrid(bp_vals, chol_vals)
        Z = np.zeros_like(BP)
 
        for i in range(BP.shape[0]):
            for j in range(BP.shape[1]):
                Z[i, j] = self.infer(float(BP[i, j]), float(CHOL[i, j]), fixed_hr)["cog"]
 
        fig = plt.figure(figsize=(11, 8))
        ax  = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(BP, CHOL, Z, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_title(f"CHD Output Surface (HR fixed at {fixed_hr})")
        ax.set_xlabel("Blood Pressure")
        ax.set_ylabel("Cholesterol")
        ax.set_zlabel("CHD Level (COG)")
        fig.colorbar(surf, shrink=0.6, aspect=10)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "chd_surface_bp_chol_fixed_hr.png"), dpi=200)
        plt.close()
 
    def plot_sensitivity(self, sensitivity_df, outpath):
        plt.figure(figsize=(8, 5))
        plt.bar(sensitivity_df["input_factor"], sensitivity_df["output_span"])
        plt.title("Sensitivity Analysis by Output Span")
        plt.xlabel("Input Factor")
        plt.ylabel("Output Span of CHD (COG)")
        plt.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()
 
 
def patient_rule_table(result):
    rows = []
    for r in result["rules"]:
        rows.append({
            "Rule": r["rule_no"],
            "Consequent": r["consequent"],
            "Antecedent Degrees": ", ".join(f"{v:.4f}" for v in r["antecedent_degrees"]),
            "Firing Strength (min)": round(r["firing_strength"], 6),
            "Rule Text": r["rule_text"],
        })
    return pd.DataFrame(rows)
 
 
def patient_fuzzification_table(result):
    rows = []
    for var, d in result["fuzzified"].items():
        for label, mu in d.items():
            rows.append({
                "Variable": var,
                "Linguistic Term": label,
                "Membership Degree": round(mu, 6),
            })
    return pd.DataFrame(rows)
 
 
def compare_results(base_result, hedge_result, patient_name):
    sys = CHDFuzzyExpertSystem
    return pd.DataFrame([
        {
            "Patient": patient_name,
            "Mode": "Base",
            "COG": round(base_result["cog"], 6),
            "Sugeno": round(base_result["sugeno"], 6),
            "COG Label": sys.linguistic_label(base_result["cog"]),
            "Sugeno Label": sys.linguistic_label(base_result["sugeno"]),
        },
        {
            "Patient": patient_name,
            "Mode": "Hedge-modified",
            "COG": round(hedge_result["cog"], 6),
            "Sugeno": round(hedge_result["sugeno"], 6),
            "COG Label": sys.linguistic_label(hedge_result["cog"]),
            "Sugeno Label": sys.linguistic_label(hedge_result["sugeno"]),
        },
    ])
 
 
def save_text_summary(path, system, patient_outputs):
    with open(path, "w", encoding="utf-8") as f:
        f.write("HealthWise CHD Fuzzy Expert System - Execution Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write("Membership design:\n")
        f.write("- BP: Low trapezoid, Medium triangle, High trapezoid\n")
        f.write("- Chol: Low trapezoid, High trapezoid\n")
        f.write("- HR: Slow trapezoid, Moderate triangle, Fast trapezoid\n")
        f.write("- CHD output: Healthy trapezoid, Middle triangle, Sick trapezoid\n")
        f.write("- Mamdani AND = min, aggregation = max, defuzzification = centroid\n")
        f.write("- Sugeno uses singleton consequents: Healthy=0.75, Middle=2.0, Sick=3.25\n")
        f.write("- Hedges: Indeed Healthy = mu^2, Somewhat Sick = sqrt(mu)\n\n")
 
        for name, base, hedge in patient_outputs:
            f.write(f"{name}\n")
            f.write("-" * len(name) + "\n")
            f.write(f"Inputs: BP={base['inputs']['bp']}, Chol={base['inputs']['chol']}, HR={base['inputs']['hr']}\n")
            f.write(f"Base COG = {base['cog']:.6f} ({system.linguistic_label(base['cog'])})\n")
            f.write(f"Base Sugeno = {base['sugeno']:.6f} ({system.linguistic_label(base['sugeno'])})\n")
            f.write(f"Hedge COG = {hedge['cog']:.6f} ({system.linguistic_label(hedge['cog'])})\n")
            f.write(f"Hedge Sugeno = {hedge['sugeno']:.6f} ({system.linguistic_label(hedge['sugeno'])})\n\n")
 
 
def main():
    outdir = os.path.join(os.path.dirname(__file__), "chd_project_outputs")
    os.makedirs(outdir, exist_ok=True)
 
    system = CHDFuzzyExpertSystem()
 
    patients = {
        "Person_1": {"bp": 105, "chol": 160, "hr": 55},
        "Person_2": {"bp": 120, "chol": 195, "hr": 65},
        "Person_3": {"bp": 165, "chol": 186, "hr": 95},
    }
 
    all_summary_rows = []
    patient_outputs  = []
 
    system.plot_membership_functions(outdir)
 
    for name, values in patients.items():
        base_result  = system.infer(values["bp"], values["chol"], values["hr"], hedged=False)
        hedge_result = system.infer(values["bp"], values["chol"], values["hr"], hedged=True)
        patient_outputs.append((name, base_result, hedge_result))
 
        patient_fuzzification_table(base_result).to_csv(
            os.path.join(outdir, f"{name.lower()}_fuzzification.csv"), index=False)
        patient_rule_table(base_result).to_csv(
            os.path.join(outdir, f"{name.lower()}_rule_firing.csv"), index=False)
        compare_results(base_result, hedge_result, name).to_csv(
            os.path.join(outdir, f"{name.lower()}_comparison.csv"), index=False)
 
        system.plot_aggregated_output(
            base_result,
            title=f"{name} - Aggregated Output (Base Mamdani)",
            filename=os.path.join(outdir, f"{name.lower()}_aggregated_base.png"),
        )
        system.plot_aggregated_output(
            hedge_result,
            title=f"{name} - Aggregated Output (Hedge-Modified Mamdani)",
            filename=os.path.join(outdir, f"{name.lower()}_aggregated_hedge.png"),
        )
 
        all_summary_rows.append({
            "Patient": name,
            "BP": values["bp"],
            "Chol": values["chol"],
            "HR": values["hr"],
            "Base COG": round(base_result["cog"], 6),
            "Base Sugeno": round(base_result["sugeno"], 6),
            "Base COG Label": system.linguistic_label(base_result["cog"]),
            "Base Sugeno Label": system.linguistic_label(base_result["sugeno"]),
            "Hedge COG": round(hedge_result["cog"], 6),
            "Hedge Sugeno": round(hedge_result["sugeno"], 6),
            "Hedge COG Label": system.linguistic_label(hedge_result["cog"]),
            "Hedge Sugeno Label": system.linguistic_label(hedge_result["sugeno"]),
        })
 
    summary_df = pd.DataFrame(all_summary_rows)
    summary_df.to_csv(os.path.join(outdir, "all_patients_summary.csv"), index=False)
 
    baseline = {"bp": 165, "chol": 186, "hr": 95}
    sens_df  = system.sensitivity_analysis(baseline=baseline, steps=150)
    sens_df.to_csv(os.path.join(outdir, "sensitivity_analysis.csv"), index=False)
    system.plot_sensitivity(sens_df, os.path.join(outdir, "sensitivity_analysis.png"))
 
    system.plot_surface(outdir, fixed_hr=95.0)
    save_text_summary(os.path.join(outdir, "execution_summary.txt"), system, patient_outputs)
 
    print("\nFinal crisp CHD values")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print("\nSensitivity analysis")
    print("=" * 60)
    print(sens_df.to_string(index=False))
    print(f"\nAll outputs saved to: {outdir}")
 
 
if __name__ == "__main__":
    main()