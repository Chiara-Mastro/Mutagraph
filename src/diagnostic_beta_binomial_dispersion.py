#!/usr/bin/env python3
"""Systematic beta-binomial audit for AMR prediction tables.

Consumes one or more prediction CSVs and checks the common parameterisation
alpha=p*phi, beta=(1-p)*phi, rho=1/(phi+1). It exports row-level parameters,
coverage, randomized PIT, standardized residuals, phi sensitivity, numerical
sanity tests, and a synthetic recovery study.

It never silently estimates phi on evaluation outcomes. For variational models
it audits the beta-binomial component conditional on the exported point prediction; 
auditing the full latent mixture requires exporting the original latent probability samples.
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import betaln, gammaln
from scipy.stats import betabinom, kstest

EPS = 1e-8
ALIASES = {
    "p": ["p_pred", "pred_p", "pred_prob", "prediction"],
    "obs": ["prop_S", "observed_prop_S", "y"],
    "ns": ["n_S", "n_s", "susceptible_count"],
    "nt": ["n_total", "n_tests", "total_count"],
    "phi": ["phi_train", "phi", "bb_phi"],
    "rho": ["rho_train", "rho", "bb_rho"],
    "model": ["model_name", "method", "model"],
    "source": ["phi_source", "dispersion_source"],
}
DEFAULT_GROUPS = [
    "Country", "target_year", "Year", "fold", "baseline_source",
    "species_seen_in_train", "family_seen_in_train",
    "species_family_seen_in_train", "both_entities_seen_in_train",
]


def named_path(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=PATH.")
    name, raw = value.split("=", 1)
    if not name.strip() or not raw.strip():
        raise argparse.ArgumentTypeError("NAME and PATH must be non-empty.")
    return name.strip(), Path(raw).expanduser()


def args_parser():
    p = argparse.ArgumentParser(description="Audit beta-binomial dispersion.")
    p.add_argument("--predictions", nargs="+", required=True, type=named_path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--coverage-levels", nargs="+", type=float,
                   default=[0.50, 0.80, 0.90, 0.95])
    p.add_argument("--phi-multipliers", nargs="+", type=float,
                   default=[0.5, 1.0, 2.0])
    p.add_argument("--group-columns", nargs="*", default=DEFAULT_GROUPS)
    p.add_argument("--pit-replicates-per-row", type=int, default=1)
    p.add_argument("--recovery-phi-values", nargs="+", type=float,
                   default=[5, 10, 25, 50, 100, 500])
    p.add_argument("--recovery-replicates", type=int, default=100)
    p.add_argument("--recovery-cells", type=int, default=1000)
    p.add_argument("--skip-recovery", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def first(cols, names):
    return next((x for x in names if x in cols), None)


def require(df, key, label):
    value = first(df.columns, ALIASES[key])
    if value is None:
        raise ValueError(f"{label}: missing {key}; tried {ALIASES[key]}")
    return value


def standardize(label: str, path: Path) :
    df = pd.read_csv(path)
    pc, nsc, ntc = require(df, "p", label), require(df, "ns", label), require(df, "nt", label)
    oc, phic, rhoc = first(df.columns, ALIASES["obs"]), first(df.columns, ALIASES["phi"]), first(df.columns, ALIASES["rho"])
    mc, sc = first(df.columns, ALIASES["model"]), first(df.columns, ALIASES["source"])
    if phic is None and rhoc is None:
        raise ValueError(f"{label}: no prospectively exported phi or rho. Refusing post-hoc test fitting.")

    out = df.copy()
    out["source_file"] = label
    out["model_audit"] = label if mc is None else label + ":" + out[mc].astype(str)
    out["p_audit"] = pd.to_numeric(out[pc], errors="coerce")
    out["n_S_audit"] = pd.to_numeric(out[nsc], errors="coerce")
    out["n_total_audit"] = pd.to_numeric(out[ntc], errors="coerce")
    out["prop_S_audit"] = (out["n_S_audit"] / out["n_total_audit"] if oc is None
                           else pd.to_numeric(out[oc], errors="coerce"))
    if phic is not None:
        out["phi_audit"] = pd.to_numeric(out[phic], errors="coerce")
    else:
        rho = pd.to_numeric(out[rhoc], errors="coerce")
        out["phi_audit"] = 1.0 / rho - 1.0
    out["rho_input"] = pd.to_numeric(out[rhoc], errors="coerce") if rhoc else np.nan
    out["phi_source_audit"] = out[sc].astype(str) if sc else "not_exported"

    out = out.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["p_audit", "n_S_audit", "n_total_audit", "prop_S_audit", "phi_audit"])
    valid = (out["n_total_audit"].gt(0) & out["n_S_audit"].ge(0)
             & out["n_S_audit"].le(out["n_total_audit"])
             & out["p_audit"].between(0, 1) & out["phi_audit"].gt(0))
    if not valid.all():
        raise ValueError(f"{label}: invalid counts, probabilities, or phi values.")
    out["n_S_audit"] = np.rint(out["n_S_audit"]).astype(int)
    out["n_total_audit"] = np.rint(out["n_total_audit"]).astype(int)
    out["p_audit"] = np.clip(out["p_audit"], EPS, 1-EPS)
    out["rho_audit"] = 1.0 / (out["phi_audit"] + 1.0)
    out["bb_alpha"] = out["p_audit"] * out["phi_audit"]
    out["bb_beta"] = (1.0 - out["p_audit"]) * out["phi_audit"]
    out["rho_abs_diff"] = (out["rho_input"] - out["rho_audit"]).abs()
    out["prop_from_counts_abs_diff"] = (out["n_S_audit"] / out["n_total_audit"] - out["prop_S_audit"]).abs()
    return out


def logpmf(k, n, p, phi):
    p, phi = np.clip(np.asarray(p, float), EPS, 1-EPS), np.maximum(np.asarray(phi, float), EPS)
    k, n = np.asarray(k, float), np.asarray(n, float)
    a, b = p*phi, (1-p)*phi
    return gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)+betaln(k+a,n-k+b)-betaln(a,b)


def add_diagnostics(df: pd.DataFrame, levels: Sequence[float], rng, pit_reps: int):
    out = df.copy()
    n, k = out.n_total_audit.to_numpy(int), out.n_S_audit.to_numpy(int)
    p, phi = out.p_audit.to_numpy(float), out.phi_audit.to_numpy(float)
    a, b = p*phi, (1-p)*phi
    out["bb_nll"] = -logpmf(k,n,p,phi)
    out["binomial_nll"] = -(gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)+k*np.log(p)+(n-k)*np.log1p(-p))
    var_count = n*p*(1-p)*(n+phi)/(1+phi)
    out["bb_variance_prop"] = var_count / n**2
    out["standardized_residual"] = (out.prop_S_audit-out.p_audit)/np.sqrt(np.maximum(out.bb_variance_prop, EPS))
    for level in levels:
        tag = str(int(round(100*level)))
        q = (1-level)/2
        lo, hi = betabinom.ppf(q,n,a,b).astype(int), betabinom.ppf(1-q,n,a,b).astype(int)
        out[f"covered_{tag}"] = (k>=lo)&(k<=hi)
        out[f"width_{tag}"] = (hi-lo)/n
        out[f"lower_prop_{tag}"], out[f"upper_prop_{tag}"] = lo/n, hi/n
    pit_parts=[]
    cdf0, mass = betabinom.cdf(k-1,n,a,b), betabinom.pmf(k,n,a,b)
    keep=["source_file","model_audit"]+[c for c in DEFAULT_GROUPS if c in out.columns]
    for r in range(pit_reps):
        part=out[keep].copy(); part["row_id_audit"]=out.index
        part["pit_replicate"]=r; part["randomized_pit"]=cdf0+rng.uniform(size=len(out))*mass
        pit_parts.append(part)
    return out, pd.concat(pit_parts, ignore_index=True)


def add_bins(df):
    out=df.copy()
    out["n_total_bin"] = pd.cut(out.n_total_audit, [0,20,50,100,250,np.inf],
                                labels=["<=20","21-50","51-100","101-250",">250"], include_lowest=True)
    out["p_pred_bin"] = pd.cut(out.p_audit, [0,.1,.25,.5,.75,.9,1.0000001],
                               labels=["[0,.1)","[.1,.25)","[.25,.5)","[.5,.75)","[.75,.9)","[.9,1]"], include_lowest=True, right=False)
    return out


def model_summary(df, pit, levels):
    rows=[]
    for model,g in df.groupby("model_audit", sort=True):
        u=pit.loc[pit.model_audit.eq(model),"randomized_pit"].dropna(); ks=kstest(u,"uniform")
        row={"model_name":model,"n_cells":len(g),"n_tests":int(g.n_total_audit.sum()),
             "phi_median":g.phi_audit.median(),"phi_min":g.phi_audit.min(),"phi_max":g.phi_audit.max(),
             "rho_mean":g.rho_audit.mean(),"bb_nll_per_cell":g.bb_nll.mean(),
             "bb_nll_per_test":g.bb_nll.sum()/g.n_total_audit.sum(),
             "binomial_nll_per_test":g.binomial_nll.sum()/g.n_total_audit.sum(),
             "std_resid_mean":g.standardized_residual.mean(),"std_resid_sd":g.standardized_residual.std(),
             "pit_mean":u.mean(),"pit_variance":u.var(),"pit_ks_statistic":ks.statistic,
             "pit_ks_p_descriptive":ks.pvalue}
        for level in levels:
            tag=str(int(round(100*level))); row[f"coverage_{tag}"]=g[f"covered_{tag}"].mean(); row[f"width_{tag}"]=g[f"width_{tag}"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_summary(df, levels, groups):
    rows=[]; dimensions=[None,"n_total_bin","p_pred_bin"]+[c for c in groups if c in df.columns]
    for dimension in dict.fromkeys(dimensions):
        group_cols=["model_audit"] if dimension is None else ["model_audit",dimension]
        for keys,g in df.groupby(group_cols, observed=True, dropna=False, sort=True):
            if not isinstance(keys,tuple): keys=(keys,)
            base={"model_name":keys[0],"stratification":"overall" if dimension is None else dimension,
                  "stratum":"all" if dimension is None else keys[1],"n_cells":len(g),"n_tests":int(g.n_total_audit.sum())}
            for level in levels:
                tag=str(int(round(100*level))); rows.append({**base,"nominal_level":level,
                    "empirical_coverage":g[f"covered_{tag}"].mean(),"coverage_error":g[f"covered_{tag}"].mean()-level,
                    "mean_interval_width":g[f"width_{tag}"].mean()})
    return pd.DataFrame(rows)


def pit_summary(pit, groups):
    rows=[]; dimensions=[None,"n_total_bin","p_pred_bin"]+[c for c in groups if c in pit.columns]
    for dimension in dict.fromkeys(dimensions):
        cols=["model_audit"] if dimension is None else ["model_audit",dimension]
        for keys,g in pit.groupby(cols, observed=True, dropna=False, sort=True):
            if not isinstance(keys,tuple): keys=(keys,)
            u=g.randomized_pit.dropna(); ks=kstest(u,"uniform")
            rows.append({"model_name":keys[0],"stratification":"overall" if dimension is None else dimension,
                         "stratum":"all" if dimension is None else keys[1],"n":len(u),"pit_mean":u.mean(),
                         "pit_variance":u.var(),"pit_q05":u.quantile(.05),"pit_median":u.median(),"pit_q95":u.quantile(.95),
                         "ks_statistic":ks.statistic,"ks_p_descriptive":ks.pvalue})
    return pd.DataFrame(rows)


def sensitivity(df, levels, multipliers):
    rows=[]
    for model,g in df.groupby("model_audit", sort=True):
        n,k,p,base=g.n_total_audit.to_numpy(int),g.n_S_audit.to_numpy(int),g.p_audit.to_numpy(float),g.phi_audit.to_numpy(float)
        for m in multipliers:
            phi=base*m; row={"model_name":model,"phi_multiplier":m,"phi_median":np.median(phi),
                "rho_median":np.median(1/(phi+1)),"bb_nll_per_test":-logpmf(k,n,p,phi).sum()/n.sum()}
            for level in levels:
                tag=str(int(round(100*level))); q=(1-level)/2; a,b=p*phi,(1-p)*phi
                lo,hi=betabinom.ppf(q,n,a,b),betabinom.ppf(1-q,n,a,b)
                row[f"coverage_{tag}"]=np.mean((k>=lo)&(k<=hi)); row[f"width_{tag}"]=np.mean((hi-lo)/n)
            rows.append(row)
    return pd.DataFrame(rows)


def fit_phi(p,k,n):
    def obj(logphi): return float(-logpmf(k,n,p,math.exp(logphi)).sum())
    result=minimize_scalar(obj,bounds=(math.log(1e-3),math.log(1e7)),method="bounded")
    return math.exp(result.x), bool(result.success)


def recovery(args, rng):
    detail=[]
    scenarios={"small":(5,30),"medium":(30,100),"large":(100,500)}
    for scenario in ["small","medium","large","heterogeneous"]:
        for true_phi in args.recovery_phi_values:
            for rep in range(args.recovery_replicates):
                if scenario=="heterogeneous": n=np.clip(np.rint(np.exp(rng.normal(math.log(70),1,args.recovery_cells))),5,1000).astype(int)
                else: lo,hi=scenarios[scenario]; n=rng.integers(lo,hi+1,args.recovery_cells)
                p=rng.beta(1.5,1.5,args.recovery_cells); theta=rng.beta(p*true_phi,(1-p)*true_phi); k=rng.binomial(n,theta)
                hat,ok=fit_phi(p,k,n); detail.append({"scenario":scenario,"phi_true":true_phi,"replicate":rep,
                    "phi_hat":hat,"rho_true":1/(true_phi+1),"rho_hat":1/(hat+1),"relative_error":(hat-true_phi)/true_phi,
                    "absolute_log_error":abs(math.log(hat)-math.log(true_phi)),"optimization_success":ok})
    d=pd.DataFrame(detail)
    s=d.groupby(["scenario","phi_true"],as_index=False).agg(n_replicates=("replicate","size"),phi_hat_median=("phi_hat","median"),
        phi_hat_mean=("phi_hat","mean"),phi_hat_sd=("phi_hat","std"),median_relative_error=("relative_error","median"),
        median_absolute_relative_error=("relative_error",lambda x:np.median(np.abs(x))),median_absolute_log_error=("absolute_log_error","median"),
        optimization_success_rate=("optimization_success","mean"))
    return d,s


def sanity_tests():
    rows=[]
    for n in [1,5,20]:
        for p in [.1,.5,.9]:
            for phi in [2,10,100]:
                k=np.arange(n+1); pmf=np.exp(logpmf(k,np.full(n+1,n),np.full(n+1,p),np.full(n+1,phi)))
                rows.append({"test":"normalization","n":n,"p":p,"phi":phi,"error":abs(pmf.sum()-1),"passed":abs(pmf.sum()-1)<1e-10})
                if p==.5: rows.append({"test":"symmetry","n":n,"p":p,"phi":phi,"error":np.max(abs(pmf-pmf[::-1])),"passed":np.max(abs(pmf-pmf[::-1]))<1e-10})
    for n in [10,100]:
        for p in [.2,.5,.8]:
            ph=np.array([2.,10.,100.,1000.]); v=n*p*(1-p)*(n+ph)/(1+ph)
            rows.append({"test":"variance_decreases_with_phi","n":n,"p":p,"phi":np.nan,"error":np.nan,"passed":bool(np.all(np.diff(v)<0))})
    return pd.DataFrame(rows)


def savefig(fig, stem, dpi):
    fig.tight_layout(); fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight"); fig.savefig(stem.with_suffix(".png"),dpi=dpi,bbox_inches="tight"); plt.close(fig)


def figures(df,pit,cov,sens,rec,out,dpi):
    data=cov[cov.stratification.eq("overall")]; fig,ax=plt.subplots(figsize=(7,5))
    for name,g in data.groupby("model_name"): ax.plot(g.nominal_level,g.empirical_coverage,marker="o",label=name)
    ax.plot([0,1],[0,1],linestyle="--",label="Ideal"); ax.set(xlabel="Nominal coverage",ylabel="Empirical coverage",title="Beta-binomial coverage calibration"); ax.legend(fontsize=7); savefig(fig,out/"beta_binomial_coverage_calibration",dpi)
    fig,ax=plt.subplots(figsize=(7,5))
    for name,g in sens.groupby("model_name"): ax.plot(g.phi_multiplier,g.bb_nll_per_test,marker="o",label=name)
    ax.set_xscale("log"); ax.set(xlabel="Phi multiplier",ylabel="NLL per test",title="Likelihood sensitivity to phi"); ax.legend(fontsize=7); savefig(fig,out/"beta_binomial_phi_sensitivity",dpi)
    fig,ax=plt.subplots(figsize=(7,5))
    for name,g in df.groupby("model_audit"): ax.scatter(g.n_total_audit,g.standardized_residual,s=4,alpha=.15,label=name)
    ax.set_xscale("log"); ax.axhline(0,linestyle="--"); ax.set(xlabel="n_total",ylabel="Standardized residual",title="Beta-binomial standardized residuals"); ax.legend(fontsize=6); savefig(fig,out/"beta_binomial_standardized_residuals",dpi)
    fig,ax=plt.subplots(figsize=(7,5))
    for name,g in pit.groupby("model_audit"): ax.hist(g.randomized_pit,bins=np.linspace(0,1,21),density=True,histtype="step",label=name)
    ax.axhline(1,linestyle="--"); ax.set(xlabel="Randomized PIT",ylabel="Density",title="Beta-binomial PIT diagnostics"); ax.legend(fontsize=7); savefig(fig,out/"beta_binomial_pit",dpi)
    if not rec.empty:
        fig,ax=plt.subplots(figsize=(7,5))
        for scenario,g in rec.groupby("scenario"): ax.plot(g.phi_true,g.phi_hat_median,marker="o",label=scenario)
        vals=rec[["phi_true","phi_hat_median"]].to_numpy().ravel(); lo,hi=np.nanmin(vals),np.nanmax(vals); ax.plot([lo,hi],[lo,hi],linestyle="--",label="Ideal")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set(xlabel="True phi",ylabel="Median estimated phi",title="Synthetic phi recovery"); ax.legend(); savefig(fig,out/"beta_binomial_phi_recovery",dpi)


def main():
    args=args_parser()
    for _,p in args.predictions:
        if not p.exists(): raise FileNotFoundError(p)
    if any(not 0<x<1 for x in args.coverage_levels): raise ValueError("Coverage levels must be in (0,1).")
    if any(x<=0 for x in args.phi_multipliers): raise ValueError("Phi multipliers must be positive.")
    args.output_dir.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(args.seed)
    df=pd.concat([standardize(n,p) for n,p in args.predictions],ignore_index=True,sort=False); df=add_bins(df)
    df,pit=add_diagnostics(df,args.coverage_levels,rng,args.pit_replicates_per_row)
    # propagate bins to PIT using row id
    bins=df[["n_total_bin","p_pred_bin"]].copy(); bins["row_id_audit"]=df.index
    pit=pit.merge(bins,on="row_id_audit",how="left",validate="many_to_one")
    summary=model_summary(df,pit,args.coverage_levels); cov=coverage_summary(df,args.coverage_levels,args.group_columns)
    pits=pit_summary(pit,args.group_columns); sens=sensitivity(df,args.coverage_levels,args.phi_multipliers); sanity=sanity_tests()
    if args.skip_recovery: rec_detail,rec_summary=pd.DataFrame(),pd.DataFrame()
    else: rec_detail,rec_summary=recovery(args,rng)
    consistency=df.groupby("model_audit",as_index=False).agg(n_rows=("p_audit","size"),n_unique_phi=("phi_audit","nunique"),phi_min=("phi_audit","min"),phi_median=("phi_audit","median"),phi_max=("phi_audit","max"),max_rho_abs_diff=("rho_abs_diff","max"),max_prop_from_counts_abs_diff=("prop_from_counts_abs_diff","max"))
    outputs={
      "parameter_export":"beta_binomial_parameter_export.csv","model_summary":"beta_binomial_model_summary.csv",
      "parameter_consistency":"beta_binomial_parameter_consistency.csv","coverage":"beta_binomial_coverage_summary.csv",
      "pit_values":"beta_binomial_pit_values.csv","pit_summary":"beta_binomial_pit_summary.csv",
      "phi_sensitivity":"beta_binomial_phi_sensitivity.csv","sanity":"beta_binomial_numerical_sanity_tests.csv",
      "recovery_detail":"beta_binomial_recovery_simulation.csv","recovery_summary":"beta_binomial_recovery_summary.csv"}
    tables={"parameter_export":df,"model_summary":summary,"parameter_consistency":consistency,"coverage":cov,"pit_values":pit,"pit_summary":pits,"phi_sensitivity":sens,"sanity":sanity,"recovery_detail":rec_detail,"recovery_summary":rec_summary}
    for key,name in outputs.items(): tables[key].to_csv(args.output_dir/name,index=False)
    figures(df,pit,cov,sens,rec_summary,args.output_dir,args.dpi)
    meta={"script":"16_audit_beta_binomial_dispersion.py","parameterization":{"alpha":"p*phi","beta":"(1-p)*phi","rho":"1/(phi+1)"},
          "conditional_on_point_prediction":True,"sanity_tests_passed":bool(sanity.passed.all()),"coverage_levels":args.coverage_levels,
          "phi_multipliers":args.phi_multipliers,"models":sorted(df.model_audit.unique()),"outputs":outputs}
    (args.output_dir/"beta_binomial_audit_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print("\nBeta-binomial audit summary\n---------------------------")
    print(summary.to_string(index=False)); print("\nSanity tests:","PASS" if sanity.passed.all() else "FAIL")
    print("\nSaved in",args.output_dir)

if __name__=="__main__": main()
