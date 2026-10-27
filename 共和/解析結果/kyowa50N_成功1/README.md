# 共和 kyowa50N_成功1 解析結果

解析手順の正本 `研究方針/解析手順２.pdf` に沿って、CH3を対象に作成した結果です。サンプリング周波数は10,000 Hz、打撃検出時刻は0.1141 s、推定モード数は8です。

- `01_output_waveform.png`：実測出力波形
- `02_fft_spectrum.png`：FFT振幅スペクトル
- `03_estimated_inputs.png`：最高使用周波数5条件の入力特解比較
- `identified_modes.csv`、`transfer_function.txt`：モードと伝達関数
- `estimated_inputs.csv`、`estimated_input_statistics.csv`：入力推定値と最大・最小・ピーク時刻
- `metadata.txt`、`fft_spectrum.csv`：条件とFFT数値

入力は単位インパルス近似・未校正の相対単位です。N（ニュートン）とは表記していません。高い最高使用周波数では伝達関数の逆演算により値が大きくなるため、物理的妥当性は校正値・ハンマー入力・追加測定との照合が必要です。
