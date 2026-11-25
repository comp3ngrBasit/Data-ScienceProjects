"""Bayesian regression of daily demand on weather + calendar (PyMC / ArviZ).

    load_t ~ StudentT(nu, mu_t, sigma)
    mu_t   = alpha + X_t @ beta            (X standardised on the training block)
    alpha  ~ Normal(mean(y_train), 2 * sd(y_train))
    beta   ~ Normal(0, sd(y_train))         weakly informative: a 1-sd change in a driver moves load by at most ~2 sd
    sigma  ~ HalfNormal(sd(y_train))
    nu     ~ Gamma(2, 0.1)                  robust to holiday outliers

Posterior predictive draws on the held-out block give probabilistic demand forecasts (credible intervals).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import RANDOM_SEED, SETTINGS
from app.utils.helpers import save_table
from app.utils.logging import get_logger

log = get_logger("bayes")


def run(df: pd.DataFrame, split: pd.Timestamp, weather_vars: list[str]) -> dict:
    import arviz as az
    import pymc as pm

    from app.analytics.weather_analysis import design

    tr, te = df.loc[:split], df.loc[split:].iloc[1:]
    Xtr, Xte = design(tr, weather_vars), design(te, weather_vars)
    mu_x, sd_x = Xtr.mean(), Xtr.std().replace(0, 1)
    Ztr, Zte = ((Xtr - mu_x) / sd_x).to_numpy(), ((Xte - mu_x) / sd_x).to_numpy()
    y = tr["load_gwh"].to_numpy()
    ym, ys = float(y.mean()), float(y.std())
    names = list(Xtr.columns)
    cfg = SETTINGS.bayes
    with pm.Model(coords={"feature": names}):
        X = pm.Data("X", Ztr)
        alpha = pm.Normal("alpha", ym, 2 * ys)
        beta = pm.Normal("beta", 0, ys, dims="feature")
        sigma = pm.HalfNormal("sigma", ys)
        nu = pm.Gamma("nu", 2, 0.1)
        mu = alpha + pm.math.dot(X, beta)
        pm.StudentT("load", nu=nu, mu=mu, sigma=sigma, observed=y, shape=X.shape[0])
        prior = pm.sample_prior_predictive(draws=1000, random_seed=RANDOM_SEED)
        idata = pm.sample(draws=cfg.draws, tune=cfg.tune, chains=cfg.chains, cores=min(cfg.chains, 2),
                          target_accept=cfg.target_accept, random_seed=RANDOM_SEED, progressbar=False)
        pm.set_data({"X": Zte})
        ppc = pm.sample_posterior_predictive(idata, var_names=["load"], random_seed=RANDOM_SEED, progressbar=False)
    summ = az.summary(idata, var_names=["alpha", "beta", "sigma", "nu"], ci_prob=0.95, ci_kind="hdi", round_to="none")
    summ = summ.reset_index(names="parameter")
    rhat_col = next(c for c in summ.columns if c.replace("_", "").lower() == "rhat")
    ess_col = next(c for c in summ.columns if "ess" in c.lower() and "bulk" in c.lower())
    # coefficients back on the original scale: GWh per unit of the driver
    post_beta = idata.posterior["beta"].stack(s=("chain", "draw")).to_numpy()  # (feature, samples)
    raw = post_beta / sd_x.to_numpy()[:, None]
    coef = pd.DataFrame({"feature": names, "mean_gwh_per_unit": raw.mean(1),
                         "ci95_low": np.quantile(raw, 0.025, 1), "ci95_high": np.quantile(raw, 0.975, 1),  # equal-tailed credible interval
                         "prob_positive": (raw > 0).mean(1)})
    # prior vs posterior samples for the key weather effects
    pr = prior.prior["beta"].stack(s=("chain", "draw")).to_numpy() / sd_x.to_numpy()[:, None]
    keep = [i for i, n in enumerate(names) if not n.startswith("dow_")]
    pp_rows = []
    for i in keep:
        pp_rows.append(pd.DataFrame({"feature": names[i], "distribution": "prior", "value": pr[i]}))
        pp_rows.append(pd.DataFrame({"feature": names[i], "distribution": "posterior", "value": raw[i]}))
    samples = pd.concat(pp_rows, ignore_index=True)
    draws = ppc.posterior_predictive["load"].stack(s=("chain", "draw")).to_numpy()  # (n_test, samples)
    fc = pd.DataFrame({"actual": te["load_gwh"].to_numpy(), "mean": draws.mean(1),
                       "lo90": np.quantile(draws, 0.05, 1), "hi90": np.quantile(draws, 0.95, 1),
                       "lo50": np.quantile(draws, 0.25, 1), "hi50": np.quantile(draws, 0.75, 1)}, index=te.index)
    cov90 = float(((fc["actual"] >= fc["lo90"]) & (fc["actual"] <= fc["hi90"])).mean() * 100)
    cov50 = float(((fc["actual"] >= fc["lo50"]) & (fc["actual"] <= fc["hi50"])).mean() * 100)
    prior_y = prior.prior_predictive["load"].stack(s=("chain", "draw")).to_numpy().ravel()
    rng = np.random.default_rng(RANDOM_SEED)
    prior_pred = pd.DataFrame({"value": rng.choice(prior_y, 20000, replace=False)})
    save_table(summ, "bayesian", "posterior_summary")
    save_table(coef, "bayesian", "coefficients")
    save_table(samples, "bayesian", "prior_posterior_samples")
    save_table(fc, "bayesian", "posterior_predictive_test")
    save_table(prior_pred, "bayesian", "prior_predictive_load")
    diverg = int(idata.sample_stats["diverging"].sum())
    log.info("Bayesian regression: max r_hat %.3f, %d divergences", summ[rhat_col].max(), diverg)
    return {"spec": __doc__.strip(), "draws": cfg.draws, "tune": cfg.tune, "chains": cfg.chains,
            "max_r_hat": float(summ[rhat_col].max()), "min_ess_bulk": float(summ[ess_col].min()), "divergences": diverg,
            "coverage_90": cov90, "coverage_50": cov50, "posterior_mean_test": fc["mean"].to_numpy(),
            "observed_train_mean_gwh": ym}
