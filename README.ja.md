# MobileSAM to ONNX

[English](README.md) | 日本語

MobileSAM（Mobile Segment Anything Model）の学習済み重みファイル（`.pt`）を、クロスプラットフォームで高速推論可能な **2つのONNXファイル（画像エンコーダ・マスクデコーダ）** へ変換（エクスポート）するためのPythonスクリプトです。

---

## 動作確認済み環境

本スクリプトおよびエクスポートされたONNXモデルは、以下のバージョン環境において動作確認を行っています。

| パッケージ / ツール             | バージョン                   |
| :------------------------------ | :--------------------------- |
| **Python**                      | `3.10.11`                    |
| **PyTorch** (`torch`)           | `2.0.1`                      |
| **TorchVision** (`torchvision`) | `0.15.2`                     |
| **ONNX** (`onnx`)               | `1.13.0`                     |
| **NumPy** (`numpy`)             | `<2` （バージョン 2.0 未満） |

---

## 前提準備 (Setup)

1. **MobileSAMパッケージの配置**
   本スクリプトを実行するには、公式の `MobileSAM/mobile_sam` ディレクトリ（パッケージ）がプロジェクト内に存在している必要があります。[MobileSAM公式リポジトリ](https://github.com/chaoningzhang/mobilesam) からソースコード一式をクローンまたはダウンロードし、作業ディレクトリ内に配置してください。

2. **学習済み重みファイルの配置**
   公式から MobileSAM の重みファイル（`mobile_sam.pt`）を取得し、以下のディレクトリ構造になるように配置します。
    ```text
    project_root/
    ├── MobileSAM/                # 公式リポジトリのディレクトリ
    │   ├── mobile_sam/
    │   ├── weights/
    │   │   └── mobile_sam.pt     # 学習済み重みファイル
    │   └── LICENSE
    ├── export_mobile_sam_onnx.py # 本エクスポートスクリプト
    └── README.md
    ```

---

## 定数・設定パラメータ一覧

スクリプト冒頭で定義されている設定値の詳細は以下の通りです。運用環境やモデルの仕様に応じて一括で変更が可能です。

| 定数名                     | 設定値（デフォルト）                  | 説明                                                                      |
| :------------------------- | :------------------------------------ | :------------------------------------------------------------------------ |
| `CHECK_POINT_FILE_PATH`    | `"./MobileSAM/weights/mobile_sam.pt"` | 入力となる学習済み重みファイル（チェックポイント）のパス                  |
| `ENCODER_OUTPUT_FILE_PATH` | `"./mobile_sam_image_encoder.onnx"`   | 出力される画像エンコーダ用ONNXファイルのパス                              |
| `DECODER_OUTPUT_FILE_PATH` | `"./mobile_sam_mask_decoder.onnx"`    | 出力されるマスクデコーダ用ONNXファイルのパス                              |
| `DEVICE`                   | `"cpu"`                               | モデルのロードおよびエクスポートを実行するデバイス（`cpu` または `cuda`） |
| `MODEL_TYPE`               | `"vit_t"`                             | MobileSAMのモデルアーキテクチャの種類                                     |
| `RESOLUTION_WIDTH`         | `1024`                                | 入力画像の基準幅（ピクセル）                                              |
| `RESOLUTION_HEIGHT`        | `1024`                                | 入力画像の基準高さ（ピクセル）                                            |
| `OPSET_VERSION`            | `17`                                  | ONNXエクスポート時のOpset（演算子セット）バージョン                       |
| `SINGLE_MASK`              | `True`                                | デコーダ出力時に単一のマスクのみを返すフラグ                              |

---

## 処理フロー

スクリプトは実行時に `export_mobile_sam` 関数を呼び出し、以下の3つのステップで処理を実行します。

### 1. モデルのロード

- 指定された `CHECK_POINT_FILE_PATH` から、`MODEL_TYPE`（`vit_t`）に一致するMobileSAMモデルをインスタンス化します。
- 指定された `DEVICE` にモデルを配置し、`eval()` メソッドにより評価モードへ移行します。

### 2. 画像エンコーダ (Image Encoder) のエクスポート

- **目的:** 入力画像から特徴量マップ（`image_embeddings`）を抽出する部分を切り出します。
- **ダミー入力:** `(1, 3, RESOLUTION_HEIGHT, RESOLUTION_WIDTH)` の形状を持つランダムな浮動小数点テンソルを作成します。
- **出力:** 指定されたパスへ `image_embeddings` を出力するONNXモデルを書き出します。

### 3. マスクデコーダ (Mask Decoder) のエクスポート

- **目的:** エンコーダが抽出した特徴量とプロンプト（点や矩形など）から、最終的なセグメンテーションマスクを予測する部分を切り出します。
- **ラッパー:** 公式の `SamOnnxModel` を使用し、`SINGLE_MASK` 設定を適用してラップします。
- **ダミー入力:**
    - 画像特徴量 (`image_embeddings`)
    - プロンプト座標 (`point_coords`)
    - プロンプトのラベル (`point_labels`)
    - マスク入力 (`mask_input`)
    - マスク有無フラグ (`has_mask_input`)
    - 元画像サイズ (`orig_im_size`)
- **動的軸（Dynamic Axes）:**
    - プロンプトの点数（`num_points`）が可変（動的）に扱えるよう設定されています。
- **出力:** `masks`（予測マスク）および `iou_predictions`（IoU信頼度スコア）を出力するONNXモデルを書き出します。

---

## 使い方 (Usage)

1. 重みファイルを指定のパス（`./weights/mobile_sam.pt`）に配置します。
2. スクリプトを実行します。

```bash
python export_mobile_sam_onnx.py
```

---

## 免責事項 (Disclaimer)

本スクリプトの利用により生じたいかなる損害・トラブル・悪影響等について、作者は一切の責任を負いません。すべて自己責任においてご利用ください。

---

## ライセンス (License)

当リポジトリのスクリプト本体は [MIT License](LICENSE) の下で公開されています。

ただし、`MobileSAM/` ディレクトリ配下のコードおよびファイルは、オリジナルの [ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM) リポジトリ（Apache License 2.0）に基づいています。詳細については `MobileSAM/LICENSE` をご参照ください。
