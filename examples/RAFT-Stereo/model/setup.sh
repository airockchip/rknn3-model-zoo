#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_SCRIPTS_DIR="$SCRIPT_DIR/../python"
cd "$SCRIPT_DIR"

# ========== default vaule ==========
HEIGHT=160
WIDTH=240
PLATFORM="rk1820"
ALL_MODELS="eth3d middlebury sceneflow realtime"
PROFILE_MODE="${PROFILE_MODE:-0}"
DTYPE=""

# ========== Usage ==========
usage() {
    cat <<EOF
Usage: $(basename "$0") <command> [options]

Commands:
  get_source       Clone RAFT-Stereo repo and download models
  to_onnx          Convert PyTorch models to ONNX format
  verify           Verify ONNX models against PyTorch checkpoints
  to_rknn          Convert ONNX models to RKNN format
  all              Run pipeline: to_onnx -> verify -> to_rknn

Options:
  -h <height>      Input height (default: $HEIGHT)
  -w <width>       Input width  (default: $WIDTH)
  -p <platform>    Target RKNN platform (default: $PLATFORM or "rk1820 rk1828 rk3572")
  -m <models>      Model list, space-separated
                   (default: "$ALL_MODELS"
                   or choose any one or combination from these)
  -t <dtype>       Quantization data type for RKNN: i8, u8, fp (default: not set)
  -P               Enable profile mode for RKNN conversion
  --help           Show this help message

Examples:
  $(basename "$0") get_source
  $(basename "$0") to_onnx -h 240 -w 320 -m "eth3d realtime"
  $(basename "$0") to_rknn -p rk1820 -t i8 -P
  $(basename "$0") to_rknn -t fp
  $(basename "$0") all -h 160 -w 240 -p rk1820 -t i8
EOF
    exit "${1:-0}"
}

# ========== parse args ==========
if [ $# -lt 1 ]; then
    usage 1
fi

COMMAND="$1"
shift

while getopts "h:w:p:m:t:P-:" opt; do
    case $opt in
        h) HEIGHT="$OPTARG" ;;
        w) WIDTH="$OPTARG" ;;
        p) PLATFORM="$OPTARG" ;;
        m) ALL_MODELS="$OPTARG" ;;
        t) DTYPE="$OPTARG" ;;
        P) PROFILE_MODE=1 ;;
        -)
            case "$OPTARG" in
                help) usage 0 ;;
                *) echo "Unknown option: --$OPTARG"; usage 1 ;;
            esac
            ;;
        *) usage 1 ;;
    esac
done
shift $((OPTIND - 1))

if [ -n "$DTYPE" ]; then
    case "$DTYPE" in
        i8|u8|fp) ;;
        *)
            echo "Error: Invalid dtype '$DTYPE'. Must be one of: i8, u8, fp"
            usage 1
            ;;
    esac
fi

RUN_MODELS="${MODEL:-$ALL_MODELS}"
RESOLUTION="${HEIGHT}x${WIDTH}"

echo "============================================"
echo " Command   : $COMMAND"
echo " Resolution: $RESOLUTION"
echo " Platform  : $PLATFORM"
echo " Models    : $RUN_MODELS"
echo " Dtype     : ${DTYPE:-(default)}"
echo " Profile   : $PROFILE_MODE"
echo "============================================"

function get_source_files {
    echo ">>> Cloning RAFT-Stereo and downloading models..."
    git clone --depth 1 https://github.com/princeton-vl/RAFT-Stereo.git
    cd RAFT-Stereo && git lfs pull
    git am ../0001-ENH-Modify-model-to-optimize-performance-in-rknn.patch
    if [ -e "$SCRIPT_DIR/RAFT-Stereo/download_models.sh" ]; then
        chmod +x download_models.sh
        bash download_models.sh
    else
        mkdir -p models
        cd models
        wget https://www.dropbox.com/s/ftveifyqcomiwaq/models.zip
        unzip models.zip
        rm -f models.zip
    fi
    cd "$SCRIPT_DIR"
}

function model_arch_args {
    case "$1" in
        realtime)
            echo "--corr_levels 4 --corr_radius 4 --n_gru_layers 2 --n_downsample 3 \
                  --context_norm batch --slow_fast_gru --shared_backbone"
            ;;
        *)
            echo "--corr_levels 4 --corr_radius 4 --n_gru_layers 3 --n_downsample 2 \
                  --context_norm batch --slow_fast_gru"
            ;;
    esac
}

function model_ckpt {
    case "$1" in
        realtime) echo "$SCRIPT_DIR/RAFT-Stereo/models/raftstereo-realtime.pth" ;;
        *)        echo "$SCRIPT_DIR/RAFT-Stereo/models/raftstereo-${1}.pth" ;;
    esac
}

function convert_2_onnx_model {
    echo ">>> Converting models to ONNX (${RESOLUTION})..."
    for model in $RUN_MODELS; do
        python "$PYTHON_SCRIPTS_DIR/convert2onnx_fp.py" \
            --raft_source "$SCRIPT_DIR/RAFT-Stereo" \
            --ckpt "$(model_ckpt "$model")" \
            $(model_arch_args "$model") \
            --out_dir "$SCRIPT_DIR/onnx" \
            --save_name "raftstereo-${model}-${RESOLUTION}" \
            --height "$HEIGHT" --width "$WIDTH"

        echo ">>> Converted ${model} to ONNX (${RESOLUTION})"
    done
}

function verify_onnx_model {
    echo ">>> Verifying ONNX models (${RESOLUTION})..."
    for model in $RUN_MODELS; do
        python "$PYTHON_SCRIPTS_DIR/verify_onnx.py" \
            --raft_source "$SCRIPT_DIR/RAFT-Stereo" \
            --onnx "$SCRIPT_DIR/onnx/raftstereo-${model}-${RESOLUTION}.onnx" \
            --ckpt "$(model_ckpt "$model")" \
            --left_img xt_left.jpg --right_img xt_right.jpg \
            $(model_arch_args "$model") \
            --height "$HEIGHT" --width "$WIDTH" \
            --vis_dir "$SCRIPT_DIR/vis"
    done
    echo ">>> ONNX verification complete"
}

function convert_2_rknn_model {
    echo ">>> Converting ONNX to RKNN (platform: $PLATFORM, dtype: ${DTYPE:-(default)})..."
    RKNN_OUTPUT_DIR="$SCRIPT_DIR/rknn"
    mkdir -p "$RKNN_OUTPUT_DIR"

    PROFILE_FLAG=""
    [ "$PROFILE_MODE" = "1" ] && PROFILE_FLAG="--profile_mode"

    DTYPE_FLAG=""
    [ -n "$DTYPE" ] && DTYPE_FLAG="--dtype $DTYPE"

    for model in $RUN_MODELS; do
        python "$PYTHON_SCRIPTS_DIR/convert2rknn.py" \
            --onnx_model_path "$SCRIPT_DIR/onnx/raftstereo-${model}-${RESOLUTION}.onnx" \
            --output "${RKNN_OUTPUT_DIR}/raftstereo-${model}-${RESOLUTION}.rknn" \
            --platform "$PLATFORM" \
            $DTYPE_FLAG \
            $PROFILE_FLAG
    done

    echo "--- RKNN output files ---"
    ls -lh "${RKNN_OUTPUT_DIR}/"
    echo ">>> RKNN conversion complete"
}

case "$COMMAND" in
    get_source)
        get_source_files
        ;;
    to_onnx)
        convert_2_onnx_model
        ;;
    verify)
        verify_onnx_model
        ;;
    to_rknn)
        convert_2_rknn_model
        ;;
    all)
        convert_2_onnx_model
        verify_onnx_model
        convert_2_rknn_model
        ;;
    --help)
        usage 0
        ;;
    *)
        echo "Error: Unknown command '$COMMAND'"
        usage 1
        ;;
esac