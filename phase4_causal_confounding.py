
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(7)


def simulate_fab_data(n=5000):
  
    # ToolMaintenance: 0 = freshly maintained, 1 = overdue maintenance
    tool_overdue = RNG.binomial(1, 0.3, n)

    # ChamberTemp drifts higher when maintenance is overdue (confounder -> "decoy" var)
    chamber_temp = 450 + tool_overdue * 8 + RNG.normal(0, 3, n)

    # EtchTime is the TRUE cause, set independently of maintenance status
    etch_time = RNG.normal(60, 5, n)

    # YieldLoss driven by: overdue maintenance (confounder, direct effect)
    #                     + etch_time deviation from ideal (TRUE cause)
    #                     + noise
    # NOTE: chamber_temp has NO direct effect on yield_loss in ground truth
    ideal_etch = 60
    yield_loss = (
        tool_overdue * 4.0
        + 0.15 * (etch_time - ideal_etch) ** 2 * 0.02
        + RNG.normal(0, 1.5, n)
    )

    return pd.DataFrame({
        "tool_overdue": tool_overdue,
        "chamber_temp": chamber_temp,
        "etch_time": etch_time,
        "yield_loss": yield_loss,
    })


def naive_correlation_analysis(df):
    print("=== Naive correlation analysis (what a rushed engineer would do) ===")
    for col in ["chamber_temp", "etch_time"]:
        r, p = stats.pearsonr(df[col], df["yield_loss"])
        print(f"  corr(yield_loss, {col:12s}) = {r:+.3f}  (p={p:.4f})")
    print("  -> Naive conclusion: chamber_temp looks like the stronger, more")
    print("     significant driver of yield loss. Engineer recommends re-tuning")
    print("     chamber temperature control. This is WRONG (see ground truth below).")


def stratified_analysis(df):
    print("\n=== Stratified analysis (adjusting for the confounder) ===")
    print("Splitting by tool maintenance status removes the confounding path:")
    for status, label in [(0, "freshly maintained"), (1, "overdue maintenance")]:
        sub = df[df["tool_overdue"] == status]
        r_temp, _ = stats.pearsonr(sub["chamber_temp"], sub["yield_loss"])
        # etch_time has a nonlinear (quadratic) relationship, so compare via
        # correlation with squared deviation, which is what the ground truth uses
        dev_sq = (sub["etch_time"] - 60) ** 2
        r_etch, _ = stats.pearsonr(dev_sq, sub["yield_loss"])
        print(f"  Within '{label}' group (n={len(sub)}):")
        print(f"    corr(yield_loss, chamber_temp)          = {r_temp:+.3f}  <- should shrink toward 0")
        print(f"    corr(yield_loss, (etch_time-60)^2)      = {r_etch:+.3f}  <- should stay strong")


def regression_with_confounder_adjustment(df):
    """Simple multivariate regression including the confounder recovers
    the correct attribution: etch_time deviation matters, chamber_temp doesn't
    (once tool_overdue is in the model)."""
    import statsmodels.api as sm

    df = df.copy()
    df["etch_dev_sq"] = (df["etch_time"] - 60) ** 2

    print("\n=== Multivariate regression (correct approach) ===")
    X = sm.add_constant(df[["tool_overdue", "chamber_temp", "etch_dev_sq"]])
    model = sm.OLS(df["yield_loss"], X).fit()
    print(model.summary().tables[1])
    print("\nInterpretation: once tool_overdue is included, chamber_temp's")
    print("coefficient becomes small/insignificant, while etch_dev_sq remains")
    print("a strong, significant predictor - correctly recovering the TRUE cause.")


if __name__ == "__main__":
    df = simulate_fab_data()
    naive_correlation_analysis(df)
    stratified_analysis(df)
    try:
        regression_with_confounder_adjustment(df)
    except ImportError:
        print("\n(install statsmodels for the regression section: pip install statsmodels)")
