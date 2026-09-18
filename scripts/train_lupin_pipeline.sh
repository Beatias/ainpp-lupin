#!/usr/bin/env bash
#
# Executa em uma única chamada o currículo completo de 3 estágios do LUPIN
# (Pavlik et al., 2025) sobre o AINPP-PB-LATAM, encadeando automaticamente
# o checkpoint de cada estágio como entrada do próximo.
#
# Cada estágio continua sendo uma chamada comum a `python main.py
# task=train ...` — este script não introduz nenhum task/engine novo, só
# fixa `training.checkpoint.dir` por estágio para saber onde procurar o
# `best_model.pt` produzido pelo `EarlyStopping` (ver utils.py) antes de
# passar para o próximo.
#
# Uso:
#   bash scripts/train_lupin_pipeline.sh [overrides hydra extras, aplicados aos 3 estágios]
#
# Exemplos:
#   bash scripts/train_lupin_pipeline.sh
#   bash scripts/train_lupin_pipeline.sh training.epochs=30 dataset.train_loader.batch_size=8
#
# Para pular estágios (ex.: já tem o checkpoint do Estágio 1 e quer só
# rodar 2 e 3), exporte STAGE1_CKPT antes de chamar o script:
#   STAGE1_CKPT=/caminho/mfunet_best.pt bash scripts/train_lupin_pipeline.sh --skip-stage1

set -euo pipefail

SKIP_STAGE1=false
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--skip-stage1" ]]; then
        SKIP_STAGE1=true
    else
        ARGS+=("$arg")
    fi
done

RUN_ROOT="outputs/lupin_pipeline/$(date +%Y-%m-%d_%H-%M-%S)"
STAGE1_DIR="${RUN_ROOT}/stage1_mfunet"
STAGE2_DIR="${RUN_ROOT}/stage2_afunet"
STAGE3_DIR="${RUN_ROOT}/stage3_final"
mkdir -p "${RUN_ROOT}"

if [[ "${SKIP_STAGE1}" == "true" ]]; then
    if [[ -z "${STAGE1_CKPT:-}" ]]; then
        echo "ERRO: --skip-stage1 requer a variável de ambiente STAGE1_CKPT apontando para um mfunet_best.pt existente." >&2
        exit 1
    fi
    echo "=== [LUPIN pipeline] Estagio 1/3 pulado (usando ${STAGE1_CKPT}) ==="
else
    echo "=== [LUPIN pipeline] Estagio 1/3 - MF-U-Net (model=lupin/mfunet loss=mfunet) ==="
    python main.py task=train model=lupin/mfunet loss=mfunet \
        training.checkpoint.dir="${STAGE1_DIR}" \
        "${ARGS[@]}"
    STAGE1_CKPT="${STAGE1_DIR}/best_model.pt"
fi

if [[ ! -f "${STAGE1_CKPT}" ]]; then
    echo "ERRO: checkpoint do Estagio 1 nao encontrado em ${STAGE1_CKPT}" >&2
    exit 1
fi

echo "=== [LUPIN pipeline] Estagio 2/3 - AF-U-Net (model=lupin/afunet loss=afunet) ==="
python main.py task=train model=lupin/afunet loss=afunet \
    model.freeze_motion_field=true \
    model.pretrained_motion_field_checkpoint="${STAGE1_CKPT}" \
    training.checkpoint.dir="${STAGE2_DIR}" \
    "${ARGS[@]}"
STAGE2_CKPT="${STAGE2_DIR}/best_model.pt"

if [[ ! -f "${STAGE2_CKPT}" ]]; then
    echo "ERRO: checkpoint do Estagio 2 nao encontrado em ${STAGE2_CKPT}" >&2
    exit 1
fi

echo "=== [LUPIN pipeline] Estagio 3/3 - Finetune conjunto (model=lupin/final loss=lupin) ==="
python main.py task=train model=lupin/final loss=lupin \
    model.pretrained_motion_field_checkpoint="${STAGE1_CKPT}" \
    model.pretrained_afunet_checkpoint="${STAGE2_CKPT}" \
    training.checkpoint.dir="${STAGE3_DIR}" \
    "${ARGS[@]}"

echo "=== [LUPIN pipeline] Concluido. ==="
echo "Checkpoint Estagio 1 (mfunet): ${STAGE1_CKPT}"
echo "Checkpoint Estagio 2 (afunet): ${STAGE2_CKPT}"
echo "Checkpoint final (LUPIN):      ${STAGE3_DIR}/best_model.pt"
