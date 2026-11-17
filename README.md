[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/cupucepu/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/cupucepu/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch viewers — hourly quantile forecast

Entry for [gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01): forecasts Grand Theft Auto VI's Twitch concurrent viewers for hours 2–24 of release day (2026-11-19 UTC), five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95) per hour, from the case file's hour-1 print, minute history, and the comparable past launches shipped in the case.

## Method

`main.py` (Python 3.13, standard library only) is an empirical comparable-quantile model: for each forecast hour it fits a weighted log-log regression of viewers on each comparable launch's own hour-1 print, then takes weighted empirical quantiles of the residuals. Weights are kernels on log hour-1 scale, viewers-per-channel, unlock factor, and pre-release level. All fitting happens at run time from the comparables inside each case file; nothing is tuned to any particular case. No network, deterministic, ~0.5 s per case.

Offline validation: practice-4 mean pinball loss 0.192, leave-one-out over the 38 comparables 0.392 — both well below the house baseline (0.387 / 0.930).

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` file in the input dir and writes one `<id>.out` per case.
