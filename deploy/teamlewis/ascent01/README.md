# ascent01 and ascent02

Two NVIDIA DGX Spark nodes. DS 4.1 Flash EXL3 on TensorFold TP 2.

## Kit

Path `~/dsv41-recipe` at git `31457d0`.
Container `dsv41-1017-exl3-tf` image `tensorfold-dsv41:exl3-tp2-03d5b88f05ef`.
API on port 8888.

## Engine pin

The Mia kit still builds `bertholomus/TensorFold` at `fbbc971`.
That pin lives in `scripts/config.sh` as `TF_REF`.
A TeamLewis swap needs a kit patch that sets `TF_REPO` to `git@github.com:tolewis/TensorFold.git`
and `TF_REF` to the `lane/fleet-unified` tip.

## Serve

`./start.sh` on ascent01. Rank 1 on ascent02 first.
Stop with `./stop.sh`.
Restart only at 0 running requests and when lane `jubiku-p4` is idle.

## Unified flags

Shared with the GLM recipe style kit: `CONTEXT`, `POOL`, `PARALLEL`, `PORT=8888`.
No swap in this pass until the DS kit is repointed at the TeamLewis fork.
