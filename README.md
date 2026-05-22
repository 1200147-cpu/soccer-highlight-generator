# ⚽ Soccer Highlight Generator GUI v2

## ファイル構成

```
soccer_highlight_gui.py   ← GUIアプリ本体
SoccerHighlight.spec      ← PyInstaller設定
build.bat                 ← EXEビルドスクリプト
```

---

## EXE化手順

### 1. ファイルを同じフォルダに置く

```
build_folder/
├── soccer_highlight_gui.py
├── SoccerHighlight.spec
├── build.bat
├── yolo11n.pt          ← ultralytics が自動DLしたものをコピー
└── ffmpeg.exe          ← gyan.dev から取得（下記参照）
```

### 2. ffmpeg.exe の入手

https://www.gyan.dev/ffmpeg/builds/ から
`ffmpeg-release-essentials.zip` をダウンロード → 解凍 → `bin\ffmpeg.exe` を取り出す

### 3. バッチを実行

```
build.bat
```

完了すると `dist\SoccerHighlight\SoccerHighlight.exe` が生成されます。

---

## 配布パッケージの作り方

```
【ZIP に含めるもの】
dist\SoccerHighlight\          ← フォルダ丸ごと（中の DLL 等が必要）
  ├── SoccerHighlight.exe
  ├── ffmpeg.exe               ← ビルド後にここへコピー
  ├── yolo11n.pt               ← ビルド後にここへコピー
  └── _internal\               ← PyInstaller 生成（触らない）
```

SoccerHighlight.exe **単体では動きません**。フォルダごと ZIP にしてください。

---

## GUI 操作方法

### 動画ファイルの追加（2通り）

| 方法 | 操作 |
|------|------|
| ドラッグ＆ドロップ | MP4 ファイルをリスト欄にドロップ（複数同時可） |
| ダイアログ |「＋ ファイルを追加」ボタン → 複数選択可 |

同じファイルは重複追加されません。

### パラメータ

| 項目 | 説明 |
|------|------|
| ハイライト数 | 最終動画に含める最大シーン数 |
| 解析FPS | 映像解析の間引き（低いほど高速・精度低） |
| 前（秒） | ハイライット点から何秒前を含めるか |
| 後（秒） | ハイライト点から何秒後を含めるか |

### ボタン

- **▶ 開始** : 処理開始
- **キャンセル** : 安全に途中停止
- **ログクリア** : ログ欄をクリア
- **選択を削除** : リストで選んだファイルを除外
- **すべてクリア** : リストを全消去

---

## ffmpeg の自動検出ロジック

1. `SoccerHighlight.exe` と同じフォルダの `ffmpeg.exe` を優先使用
2. 見つからない場合は PATH から検索
3. 起動時にログ欄に検出結果を表示

---

## 注意事項

- **inputフォルダは不要** になりました。ファイルを直接指定してください。
- 処理済みファイルは archive 移動しません（元ファイルはそのまま残ります）。
- 大歓声シーン（audio ≥ 0.9）は前後秒数が自動で 2 倍になります。
- 一時ファイル（`_tmp_audio_*.wav`, `_hl_*.mp4`）は出力フォルダに生成され処理後自動削除されます。

---

## トラブルシューティング

| 症状 | 対処 |
|------|------|
| 起動しない | Microsoft Visual C++ Redistributable をインストール |
| ffmpeg エラー | `ffmpeg.exe` を EXE と同フォルダに置く |
| YOLO読み込みエラー | `yolo11n.pt` を EXE と同フォルダに置く |
| 音声抽出失敗 | MP4 に音声トラックがあるか確認 |
