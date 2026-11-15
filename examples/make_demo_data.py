"""Write SYNTHETIC demo inputs: ocean predictor table + daily station observations (long CSV format).

The rainfall / temperature relationships to the ocean indices are planted on purpose, so the demo shows what the outputs look
like when there IS skill.  Real data will give much weaker skill: treat the demo as a format and workflow tour, nothing more.

    python examples/make_demo_data.py demo_data
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    months = pd.date_range("1981-01-01", "2026-12-01", freq="MS")

    def ar(phi, sd):
        x = np.zeros(len(months)); e = rng.normal(0, sd, len(months))
        for i in range(1, len(months)):
            x[i] = phi * x[i - 1] + e[i]
        return x

    ind = pd.DataFrame({"R34": ar(0.93, 0.35), "WIOD": ar(0.85, 0.25), "EIOD": ar(0.85, 0.25),
                        "TNA": ar(0.88, 0.25), "TSA": ar(0.88, 0.25)}, index=months)
    ind["DMI"] = ind["WIOD"] - ind["EIOD"]
    ind.index.name = "Date"
    ind.round(4).reset_index().to_csv(out / "predictors.csv", index=False)

    # standardised Mar-May mean of R34 and TNA per year = what a 1 May forecast can see
    def spring(col):
        s = ind[col][ind.index.month.isin([3, 4, 5])].groupby(ind.index[ind.index.month.isin([3, 4, 5])].year).mean()
        return (s - s.mean()) / s.std()
    r34, tna = spring("R34"), spring("TNA")

    days = pd.date_range("1981-01-01", "2026-05-31")                       # observations end the day before a 1 June forecast
    yr = days.year
    stations = [("STN0001", 10.4, 35.0), ("STN0002", 11.2, 36.1), ("STN0003", 9.1, 38.7), ("STN0004", 12.6, 37.5),
                ("STN0005", 8.5, 36.9), ("STN0006", 13.5, 39.5)]
    rows = []
    for k, (stn, lat, lon) in enumerate(stations):
        sign = 1 if k % 2 == 0 else 0.6
        z = (-0.9 * sign * r34.reindex(yr).values + 0.5 * tna.reindex(yr).values)       # El Nino-like warmth dries; warm N. Atlantic wets
        wet = np.isin(days.month, [6, 7, 8, 9])
        pday = np.where(wet, expit(logit(0.45) + 0.35 * z), 0.08)
        amount = rng.gamma(2.0, 6.0 + 1.5 * k, len(days)) * np.where(wet, np.exp(0.15 * z), 1.0)
        rain = np.where(rng.random(len(days)) < pday, amount + 1.0, 0.0)
        tmax = 27 + 2 * np.sin(2 * np.pi * (days.dayofyear - 80) / 365) - 0.3 * k + np.where(wet, 0.5 * r34.reindex(yr).values, 0) + rng.normal(0, 1.5, len(days))
        for var, v in (("precipitation", rain), ("tmax", tmax)):
            rows.append(pd.DataFrame({"Station": stn, "Date": days.strftime("%Y-%m-%d"), "Lat": lat, "Lon": lon, "Variable": var, "Value": np.round(v, 2)}))
    pd.concat(rows).to_csv(out / "stations_long.csv", index=False)
    print(f"Wrote {out/'predictors.csv'} and {out/'stations_long.csv'} ({len(stations)} stations, {days[0].date()} .. {days[-1].date()})")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "demo_data"))
