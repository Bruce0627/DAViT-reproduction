# DAViT 論文復現

Fu、Tantithamthavorn、Le，IEEE Access 13 (2025)，103033–103044。[論文 DOI](https://doi.org/10.1109/ACCESS.2025.3579314) · [作者程式與日誌](https://github.com/awsm-research/DAViT/tree/021509e7d546dbf01c0ee898edd6a18debaa9051)

## 目前驗證結果

使用作者公開權重，在 NVIDIA RTX 3060（PyTorch 2.6.0、CUDA 12.4、Transformers 4.46.3、batch 8）對 Kermany 測試影像重新推論。

| 任務 | 樣本數 | TN / FP / FN / TP | Accuracy | F1 | 作者版 AUC | 機率 ROC-AUC |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 肺炎偵測 | 624 | 218 / 16 / 8 / 382 | 96.15% | 96.95% | 95.56% | 99.25% |
| 細菌型／病毒型 | 390 | 240 / 2 / 47 / 101 | 87.44% | 80.48% | 83.71% | 95.22% |

肺炎偵測的混淆矩陣及六項指標均重現。類型分類相較作者日誌少判對 1 張病毒型影像，F1 低 0.47 個百分點。已逐張確認 Kaggle 與作者公開資料夾的 624 張測試影像位元完全一致。直接呼叫作者的 `TextDataset` 與 `test` 一樣得到 F1 80.48%，改用 batch 32 或 Pillow 10.4.0 時結果也相同。原日誌沒有逐張預測，因此差異的確切原因仍未確認。逐張預測、三張 Grad-CAM、指標 JSON 和權重 SHA-256 位於 `results/`。本次驗證使用發布的權重，完整重訓、基準模型及消融實驗需另外執行。

以下為作者日誌的重算對照。使用日誌的召回率和特異度與測試集的正負類數還原混淆矩陣。六項重算值與日誌一致，亦核對了 Kaggle 各切分的樣本數與 624 張測試影像。驗證紀錄見 `results/verification.json`。

| 任務 | 樣本數 | TN / FP / FN / TP | Accuracy | Precision | Recall | Specificity | F1 | 作者版 AUC |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 肺炎偵測 | 624 | 218 / 16 / 8 / 382 | 96.15% | 95.98% | 97.95% | 93.16% | 96.95% | 95.56% |
| 細菌型／病毒型 | 390 | 240 / 2 / 46 / 102 | 87.69% | 98.08% | 68.92% | 99.17% | 80.95% | 84.05% |

作者的 `davit/main.py` 將 argmax 預測標籤交給 `roc_auc_score`。此二元 AUC 等於 (Recall + Specificity) / 2。執行權重驗證後，`auc_hard_labels` 用來對照論文，`auc_probability` 用正類 softmax 機率計算跨閾值 ROC-AUC。偵測的正類為肺炎，類型分類的正類為病毒型。

只需 Python 3 即可重算日誌數值。執行位置為解壓後根目錄。

```powershell
python reproduce.py --reference-only
```

## 重新推論

使用 Python 3.12 與可用的 CUDA 裝置，或以 CPU 執行。單卡使用預設 batch 8，記憶體有限時可調到 2。以下從解壓後根目錄執行。CPU 安裝可將 PyTorch 套件來源換成 `https://download.pytorch.org/whl/cpu`。

```powershell
git clone https://github.com/awsm-research/DAViT.git DAViT
git -C DAViT checkout 021509e7d546dbf01c0ee898edd6a18debaa9051
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

準備資料與權重。權重各約 1.26 GB，使用以下作者檔案，放入 `DAViT/davit/saved_models/checkpoint-best-f1/`。

- [davit_pneumonia_detection.bin](https://drive.google.com/file/d/17FAS8mtMFebPkXVPCWTyKYKLS4TI5zi4/view)
- [davit_pneumonia_type.bin](https://drive.google.com/file/d/1mc5wvL5FlFXoqeQs-VCpKjPIhsB9x1wB/view)

資料採 [Kermany 肺炎胸部 X 光 Kaggle 版本](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) 的原始固定切分。作者 README 指向其 `data/download_data.sh`。Kaggle 壓縮檔內含兩份目錄，取其中一份，讓 `DAViT/data/chest_xray/test` 下直接包含 `NORMAL` 和 `PNEUMONIA`。評估程式檢查各類別張數。資料夾不同時，用 `--data` 或 `--weights` 指定。

```powershell
python reproduce.py --task both
python reproduce.py --task type --cam "DAViT/data/chest_xray/test/PNEUMONIA/實際檔名.jpeg" --target 1
```

第一次推論會向 Hugging Face 取得 [DINOv2-Large](https://huggingface.co/facebook/dinov2-large) 的模型設定與前處理設定。可用 `--backbone` 指定包含 `config.json`、`preprocessor_config.json` 的本機資料夾。輸出包含每張影像的預測與正類機率 CSV、每項任務的指標 JSON，以及各指標與作者日誌的差值。可重複指定 `--cam`，用 CNN 最後一層活化與目標類別機率為選定影像產生 Grad-CAM 疊圖。疊圖採用測試流程的 256×256 尺度與紅色透明遮罩。論文圖 3 的作者筆記本使用 224×224 和多色疊圖。

## 重跑下游訓練

作者提供 [CXR8 領域適應權重](https://drive.google.com/file/d/1YAIq55dGq7_68B-UzgIUTLC2ghJevVh1/view)。將 `domain_adapted_davit.bin` 放入相同權重目錄，再在 Linux/WSL、PyTorch CUDA 環境下由作者程式執行。訓練前先保留下載的下游權重，或在另一路徑保留它們，因為原腳本會將訓練權重寫入同名檔案。

```bash
cd DAViT/davit
python -m pip install scikit-learn==1.6.1 tqdm
bash -o pipefail train_detection.sh
bash -o pipefail train_type.sh
```

使用 `train`、`val`、`test` 原切分。作者腳本採 30 epochs、batch 32、seed 123456、AdamW、學習率 10⁻⁵、線性排程含 10% warm-up，以最低驗證損失選模。論文報告最大梯度範數為 1，公開的 `davit/main.py` 註解了梯度裁切呼叫。公開程式亦在建立訓練資料集時一次性套用隨機增強。原入口寫死兩張 CUDA GPU。

套件附有 `train.py`，可在 Windows 或 Linux 從同一份作者程式呼叫訓練迴圈，預設使用單張 CUDA GPU 及較小的 batch。可用 `--gpus 2 --train-batch-size 32 --eval-batch-size 32` 採用作者設定。batch 會影響更新步數與排程，應隨實驗記錄。新權重預設放在 `training/checkpoint-best-f1`，可把它傳給 `reproduce.py --weights` 來評估。

```powershell
python -m pip install scikit-learn==1.6.1 tqdm
python train.py --task detection
python train.py --task type
python reproduce.py --weights training/checkpoint-best-f1
```

## CXR8 領域適應

作者釋出的下游入口使用已適應的權重。若有原文所述 CXR8 影像與切分，可用套件附的入口從 DINOv2-Large 開始跑。CSV 必須含 `path,split,label`，其中 `split` 為 `train` 或 `val`，`label` 為 0（正常）或 1（至少一項異常）。`path` 可相對於清單所在目錄。請按病人分組後準備切分，以免同一病人跨集合。這個入口使用作者的訓練迴圈與二元目標，逐張載入影像並在訓練時執行增強。由於原始資料清單和領域適應入口未公開驗證，重訓結果應記錄所用的清單和切分。

```powershell
python train.py --task adapt --cxr8-manifest cxr8.csv --output-dir adaptation
python train.py --task detection --domain-weights adaptation/checkpoint-best-f1/domain_adapted_davit.bin
python train.py --task type --domain-weights adaptation/checkpoint-best-f1/domain_adapted_davit.bin
```

原文表 3 需分別重訓各消融設定。套件的預設入口對應完整 DAViT。

本機用合成影像和縮小的骨幹各跑 1 個 epoch，驗證領域適應與下游偵測入口可產生權重，且下游能載入前一階段的權重。這項測試只查核程式流程，沒有使用 CXR8 或 Kermany 影像重訓。
