########################################################
# MobileSAMの学習済み重みファイルを２つのONNXファイルに変換
# - 画像エンコーダONNXファイル
# - マスクデコーダONNXファイル
########################################################
import warnings
from torch.jit import TracerWarning

# 指定した種類の警告だけを非表示（無視）にする
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=TracerWarning)

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "MobileSAM"))

import torch
from MobileSAM.mobile_sam import sam_model_registry
from MobileSAM.mobile_sam.utils.onnx import SamOnnxModel

# 学習済み重みファイル（チェックポイント）
CHECK_POINT_FILE_PATH = "./MobileSAM/weights/mobile_sam.pt"
# 画像エンコーダONNXファイル
ENCODER_OUTPUT_FILE_PATH = "./mobile_sam_image_encoder.onnx"
# マスクデコーダONNXファイル
DECODER_OUTPUT_FILE_PATH = "./mobile_sam_mask_decoder.onnx"
# 実行デバイス
DEVICE = "cpu"
# モデルアーキテクチャ
MODEL_TYPE = "vit_t"
# エクスポート用画像解像度
RESOLUTION_WIDTH = 1024
RESOLUTION_HEIGHT = 1024
# Opsetバージョン
OPSET_VERSION = 17
# マスクデコーダの動作フラグ（True: 単一のマスク）
SINGLE_MASK = True

def export_mobile_sam(checkpoint_path: str):
    # ==========================================
    # 学習済み重みファイルのロード
    # ==========================================
    print("1. MobileSAM モデルをロード中...")
    print(f" チェックポイントファイル: {CHECK_POINT_FILE_PATH}\n")

    # ViT-TアーキテクチャのMobileSAMモデルをチェックポイントからインスタンス化し、
    # デバイスに配置して評価モードに設定
    mobile_sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint_path)
    mobile_sam.to(DEVICE)
    mobile_sam.eval()

    # ==========================================
    # 画像エンコーダ (Image Encoder) のエクスポート
    # ==========================================
    print(f"2. 画像エンコーダをエクスポート中: {ENCODER_OUTPUT_FILE_PATH}")

    # 画像エンコーダへの入力となるダミー画像テンソルを作成
    # （バッチサイズ1、RGBの3チャンネル、1024x1024ピクセル）
    dummy_image = torch.randn(1, 3, RESOLUTION_HEIGHT, RESOLUTION_WIDTH, device=DEVICE)

    # 画像エンコーダモデルをONNX形式に変換してファイルに出力
    torch.onnx.export(
        mobile_sam.image_encoder,
        dummy_image,
        ENCODER_OUTPUT_FILE_PATH,
        input_names=["image"],
        output_names=["image_embeddings"],
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
    )
    print(" ==> 画像エンコーダのエクスポート完了。\n")

    # ==========================================
    # マスクデコーダ (Mask Decoder / SamOnnxModel) のエクスポート
    # ==========================================
    print(f"3. マスクデコーダをエクスポート中: {DECODER_OUTPUT_FILE_PATH}")
    
    # 公式の SamOnnxModel ラッパーを使用
    # （単一のマスクを返す設定でモデルをラップし、評価モードに設定）
    onnx_model = SamOnnxModel(model=mobile_sam, return_single_mask=SINGLE_MASK)
    onnx_model.eval()

    # プロンプトエンコーダから埋め込み次元数と画像埋め込みサイズ（64x64）を取得
    embed_dim = mobile_sam.prompt_encoder.embed_dim
    embed_size = mobile_sam.prompt_encoder.image_embedding_size  # (64, 64)
    
    # 公式 SamOnnxModel が期待するダミー入力のセットを定義
    dummy_inputs = (
        torch.randn(1, embed_dim, *embed_size, device=DEVICE), # image_embeddings
        torch.randint(0, RESOLUTION_WIDTH, (1, 1, 2), dtype=torch.float, device=DEVICE), # point_coords
        torch.randint(0, 4, (1, 1), dtype=torch.float, device=DEVICE), # point_labels
        torch.randn(1, 1, 256, 256, dtype=torch.float, device=DEVICE), # mask_input
        torch.tensor([0], dtype=torch.float, device=DEVICE), # has_mask_input
        torch.tensor([RESOLUTION_HEIGHT, RESOLUTION_WIDTH], dtype=torch.float, device=DEVICE), # orig_im_size
    )

    # ONNXモデルの入力ノード名を定義
    input_names = [
        "image_embeddings",
        "point_coords",
        "point_labels",
        "mask_input",
        "has_mask_input",
        "orig_im_size",
    ]
    # ONNXモデルの出力ノード名を定義
    output_names = ["masks", "iou_predictions"]

    # プロンプトの点数を可変（動的）に扱えるように動的軸を設定
    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    # マスクデコーダモデルをONNX形式に変換してファイルに出力
    torch.onnx.export(
        onnx_model,
        dummy_inputs,
        DECODER_OUTPUT_FILE_PATH,
        export_params=True,
        verbose=False,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print(" ==> マスクデコーダのエクスポート完了。\n")

if __name__ == "__main__":
    export_mobile_sam(CHECK_POINT_FILE_PATH)
