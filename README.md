# Speech Emotion Recognition Using BiLSTM

**Deep Learning-Based Emotional Speech Analysis**

This B.Tech research prototype classifies WAV recordings as Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, or Surprised. It includes a bidirectional LSTM classifier, a unidirectional LSTM baseline, a speaker-independent evaluation pipeline, and a five-page Streamlit dashboard.

```mermaid
flowchart LR
    A[WAV upload or microphone] --> B[Mono audio at 22,050 Hz]
    B --> C[Trim and fit to 3 seconds]
    C --> D[130 frames × 40 MFCCs]
    D --> E[Training-fitted standardization]
    E --> F[BiLSTM]
    F --> G[Eight emotion scores]
```

## Measured results

Both models were evaluated on the same 240 recordings from four actors held out from training and validation.

| Model | Test accuracy | Macro F1 | Trainable parameters |
| --- | ---: | ---: | ---: |
| BiLSTM | 32.08% | 0.286 | 97,320 |
| Standard LSTM | 28.75% | 0.208 | 40,616 |

These results come from a single actor split using seed 42. Run the training and evaluation commands below to reproduce the experiment and generate the checkpoints and figures locally.

## 1. Windows setup

Install 64-bit Python 3.11 and open this project folder in VS Code. Run these commands in its PowerShell terminal.

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

The execution policy change applies to the current shell. You can also run `.\.venv\Scripts\python.exe` directly. In VS Code, select `.venv\Scripts\python.exe` via **Python: Select Interpreter**.

Native Windows uses the CPU. If TensorFlow reports a missing DLL, check the Microsoft Visual C++ x64 Redistributable and Windows long-path support. For optional CUDA training on Windows, use WSL2 and the supported TensorFlow GPU installation. [TensorFlow installation guide](https://www.tensorflow.org/install/pip).

## 2. Set up RAVDESS

Download **Audio_Speech_Actors_01-24.zip** from the official record. The archive is about 208.5 MB and contains 1,440 audio-only speech WAV files from 24 actors. [RAVDESS on Zenodo](https://zenodo.org/records/1188976).

If you saved the ZIP to Downloads, run the following from the project folder. The commented command downloads the same archive from PowerShell if needed.

```powershell
$archive = Join-Path $env:USERPROFILE 'Downloads\Audio_Speech_Actors_01-24.zip'
# curl.exe -fL --retry 3 -o $archive 'https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1'
$expectedMd5 = 'BC696DF654C87FED845EB13823EDEF8A'
if ((Get-FileHash -LiteralPath $archive -Algorithm MD5).Hash -ne $expectedMd5) {
    throw 'RAVDESS speech archive checksum mismatch.'
}
Expand-Archive -LiteralPath $archive -DestinationPath .\dataset
(Get-ChildItem .\dataset -Recurse -File -Filter '*.wav' | Measure-Object).Count
```

You should see `1440`. The loader searches recursively for `Actor_01` through `Actor_24`, including within an outer extraction folder. If the archive is already extracted elsewhere, pass `--dataset-dir 'C:\path\to\ravdess'` to both training and evaluation.

## 3. Train, evaluate, and run the app

```powershell
python train.py --model bilstm
python train.py --model lstm
python evaluate.py --model both
python predict.py .\dataset\Actor_08\03-01-02-01-02-02-08.wav
streamlit run app.py
```

Adjust the WAV path for your extraction layout. `python train.py --model both` trains both architectures with one preprocessing pass. The defaults are 60 maximum epochs per model, batch size 32, learning rate 0.001, and seed 42. Use `--epochs`, `--batch-size`, `--learning-rate`, `--seed`, and `--threads` to adjust a run. Use a fresh `--output-dir` for a different actor split or dataset, and pass that directory to evaluation and CLI prediction. The Streamlit dashboard reads the project's `outputs/` directory.

For a live demo, open **Emotion Detection** at [localhost:8501](http://localhost:8501). Upload a WAV, play it back, and click **Analyze Emotion**. The page shows the predicted label, all eight softmax scores, the original waveform, and the MFCCs used for inference. You can also select **Record microphone** and allow microphone access in your browser and Windows settings. Use localhost or an HTTPS deployment for browser recording. [Streamlit audio input documentation](https://docs.streamlit.io/develop/api-reference/widgets/st.audio_input).

## Experimental protocol

- **Speaker-independent split.** The seeded split assigns 16 actors to training, four to validation, and four to testing, with balanced representation of the dataset's documented male and female actor groups. Both models use the same assignments. The manifest records each file's label, actor, partition, duration and SHA-256.
- **Audio preprocessing.** SoundFile decodes WAV data and channels are averaged to mono. Librosa resamples to 22,050 Hz, trims low-energy margins at 40 dB below peak frame RMS, and takes the first three seconds or right-pads with zeros. A centered 2,048-point FFT with a 512-sample hop yields 130 frames of 40 MFCCs. Normalization uses per-coefficient training means and standard deviations across training frames only.
- **Models.** Both architectures have recurrent widths 64 and 32, two dropout layers at 0.3, a 32-unit ReLU layer, and an eight-unit softmax. Bidirectionality increases the parameter count. Training uses Adam, sparse categorical cross-entropy, and accuracy. Validation loss drives checkpoint selection, early stopping, and learning-rate reduction.
- **Evaluation.** Evaluation reloads the minimum-validation-loss checkpoint and uses only held-out test actors. It reports accuracy, macro precision, macro recall, macro and weighted F1, and per-class results. Inference time is the mean warmed-up batched forward-pass time per clip and excludes feature extraction. Training time includes first-epoch tracing, validation, and checkpoint writes. Use `evaluate.py --model both` to benchmark both models together.

## Research outputs

Run training and evaluation to populate `outputs/`. The dashboard exposes downloadable files on **Research Results**.

| Stage | Files |
| --- | --- |
| Dataset validation and preprocessing | `dataset_manifest.csv`, `dataset_statistics.json`, `emotion_distribution.png`, `audio_duration_distribution.png`, `waveform.png`, `mfcc_heatmap.png` |
| Training | `bilstm_emotion.keras`, `lstm_baseline.keras`, `scaler.npz`, `baseline_scaler.npz`, `training_config.json`, `training_history.csv`, `training_accuracy.png`, `training_loss.png`, `model_summary.txt`, `model_architecture.png` |
| Held-out evaluation | `test_metrics.json`, `test_predictions.csv`, `classification_report.txt`, `classification_report.csv`, `confusion_matrix.png`, `confusion_matrix.csv`, `emotion_f1_scores.png` |
| Research summary | `research_results.md`, plus `model_comparison.csv` and `model_comparison.png` when both models have been evaluated under the same protocol |

LSTM's history, configuration, figures, and evaluation files use the `baseline_` prefix. For inference, keep `bilstm_emotion.keras`, `scaler.npz`, and `training_config.json` together. The dataset and generated model artifacts stay local and are excluded from Git.

## Limitations and citation

RAVDESS contains acted English speech recorded under controlled conditions and uses two repeated sentences. Natural speech, other languages, and noisy microphone recordings may produce unreliable results. Softmax scores are uncalibrated. A single held-out split gives a result for four test speakers, and the larger BiLSTM differs from the baseline in both directionality and parameter count.

Livingstone SR and Russo FA (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English.* PLOS ONE 13(5), e0196391. [Paper DOI](https://doi.org/10.1371/journal.pone.0196391). [Dataset record and CC BY-NC-SA 4.0 terms](https://zenodo.org/records/1188976).
