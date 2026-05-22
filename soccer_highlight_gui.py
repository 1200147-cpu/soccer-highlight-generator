"""
サッカー自動ハイライト生成 GUI版 v2
- ffmpeg を EXE と同じフォルダから自動検出
- ファイル選択 / ドラッグ&ドロップで MP4 を追加（複数可）
- PyQt6 + PyInstaller 対応
"""

import sys
import os
import shutil
import subprocess

import cv2
import numpy as np
import librosa

from datetime import datetime
from ultralytics import YOLO

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QSpinBox,
    QDoubleSpinBox, QTextEdit, QProgressBar,
    QFileDialog, QGroupBox, QFrame, QListWidget,
    QListWidgetItem, QAbstractItemView,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QTextCursor, QDragEnterEvent, QDropEvent


# ==========================================
# ffmpeg / YOLO パス解決
# ==========================================

def _base_dir() -> str:
    """EXE なら exe のフォルダ、スクリプトならスクリプトのフォルダ"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def get_ffmpeg() -> str:
    local = os.path.join(_base_dir(), "ffmpeg.exe")
    return local if os.path.isfile(local) else "ffmpeg"


def get_yolo_path() -> str:
    local = os.path.join(_base_dir(), "yolo11n.pt")
    return local if os.path.isfile(local) else "yolo11n.pt"


FFMPEG = get_ffmpeg()

# ==========================================
# subprocess ウィンドウ非表示設定（Windows用）
# ==========================================

def _get_si():
    """CMDウィンドウが一切出ないようにするSTARTUPINFO"""
    if sys.platform == "win32":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE
        return si
    return None

_SI = _get_si()

# ==========================================
# CPU最適化設定
# ==========================================

import multiprocessing
_CPU_COUNT = multiprocessing.cpu_count()

# OpenCV：全CPUコアを使う
cv2.setNumThreads(_CPU_COUNT)
cv2.setUseOptimized(True)


# ==========================================
# ワーカースレッド
# ==========================================

class HighlightWorker(QObject):

    log_signal      = pyqtSignal(str)
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(bool, str)
    step_signal     = pyqtSignal(int)

    def __init__(self, params: dict):
        super().__init__()
        self.params  = params
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def log(self, msg: str):
        self.log_signal.emit(msg)

    def run(self):
        try:
            self._process()
        except Exception:
            import traceback
            self.finished_signal.emit(False, traceback.format_exc())

    def _process(self):
        p = self.params

        video_paths      = p["video_paths"]
        OUTPUT_DIR       = p["output_dir"]
        RESIZE_WIDTH     = 480
        ANALYZE_FPS      = p["analyze_fps"]
        TOP_HIGHLIGHTS   = p["top_highlights"]
        HIGHLIGHT_BEFORE = p["highlight_before"]
        HIGHLIGHT_AFTER  = p["highlight_after"]
        MIN_SCENE_INTERVAL = max(HIGHLIGHT_BEFORE, HIGHLIGHT_AFTER) * 2

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        timestamp            = datetime.now().strftime("%Y%m%d_%H%M%S")
        FINAL_HIGHLIGHT_NAME = f"final_highlight_{timestamp}.mp4"

        self.log(f"ffmpeg : {FFMPEG}")
        self.log(f"対象動画: {len(video_paths)} ファイル")
        self.log(f"設定 → ハイライト数:{TOP_HIGHLIGHTS} / 前:{HIGHLIGHT_BEFORE}秒 / 後:{HIGHLIGHT_AFTER}秒\n")

        # YOLO ロード
        self.step_signal.emit(0)
        self.log("YOLOモデルロード中...")
        self.progress_signal.emit(2, "YOLOロード中...")
        model = YOLO(get_yolo_path())
        # CPUスレッド数を明示設定
        model.overrides["workers"] = _CPU_COUNT
        self.log(f"YOLOロード完了 (CPU {_CPU_COUNT}コア)\n")

        all_highlights = []
        total_videos   = len(video_paths)

        for video_index, VIDEO_PATH in enumerate(video_paths):

            if self._cancel:
                self.finished_signal.emit(False, "キャンセルされました")
                return

            video_name    = os.path.basename(VIDEO_PATH)
            base_progress = int(video_index / total_videos * 80)

            self.log(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            self.log(f"[{video_index+1}/{total_videos}] {video_name}")
            self.log(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            # ── 1/6 音声解析 ──
            self.step_signal.emit(1)
            self.log("【1/6】音声解析中...")
            self.progress_signal.emit(base_progress + 2,
                                      f"[{video_index+1}/{total_videos}] 音声抽出中")

            audio_path = os.path.join(OUTPUT_DIR, f"_tmp_audio_{video_index}.wav")
            subprocess.run(
                [FFMPEG, "-y", "-i", VIDEO_PATH,
                 "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1",
                 audio_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                startupinfo=_SI
            )

            if not os.path.exists(audio_path):
                self.log("  ✗ 音声抽出失敗 → スキップ")
                continue

            # mono=True・dtype=float32 で読み込み高速化
            y, sr = librosa.load(audio_path, mono=True, dtype=np.float32)

            # 1秒ごとのチャンクをまとめてnumpy処理（ループ削減）
            n_chunks     = len(y) // sr
            y_trimmed    = y[:n_chunks * sr].reshape(n_chunks, sr)
            fft_all      = np.abs(np.fft.rfft(y_trimmed, axis=1))
            freqs        = np.fft.rfftfreq(sr, d=1/sr)
            voice_mask   = (freqs > 800) & (freqs < 4000)
            audio_scores = np.mean(fft_all[:, voice_mask], axis=1).astype(np.float32)

            # 移動平均（numpy convolve で高速化）
            window       = 2
            kernel       = np.ones(window * 2 + 1) / (window * 2 + 1)
            audio_scores = np.convolve(audio_scores, kernel, mode='same').astype(np.float32)
            if np.max(audio_scores) > 0:
                audio_scores /= np.max(audio_scores)

            os.remove(audio_path)
            self.log("  ✓ 音声解析完了")

            # ── 2/6 映像解析 ──
            self.step_signal.emit(2)
            self.log("【2/6】映像解析中...")

            cap           = cv2.VideoCapture(VIDEO_PATH)
            fps           = cap.get(cv2.CAP_PROP_FPS)
            total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_seconds = int(total_frames / fps) if fps > 0 else 0
            self.log(f"  動画時間: {video_seconds//60}分{video_seconds%60}秒")

            frame_interval  = max(1, int(fps / ANALYZE_FPS))
            ret, prev_frame = cap.read()
            if not ret:
                cap.release()
                continue

            h, w   = prev_frame.shape[:2]
            new_h  = int(h * RESIZE_WIDTH / w)
            prev_frame = cv2.resize(prev_frame, (RESIZE_WIDTH, new_h))
            prev_gray  = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

            motion_scores  = []
            density_scores = []
            frame_count    = 0

            while True:
                if self._cancel:
                    cap.release()
                    self.finished_signal.emit(False, "キャンセルされました")
                    return

                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                if frame_count % max(1, frame_interval * 30) == 0:
                    prog = base_progress + 2 + int(frame_count / total_frames * 10)
                    pct  = frame_count / total_frames * 100
                    cur  = int(frame_count / fps)
                    self.progress_signal.emit(
                        prog,
                        f"[{video_index+1}/{total_videos}] 映像解析 {pct:.0f}%"
                        f" ({cur//60}:{cur%60:02d})"
                    )

                if frame_count % frame_interval != 0:
                    continue

                frame = cv2.resize(frame, (RESIZE_WIDTH, new_h))
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                motion_scores.append(np.mean(cv2.absdiff(prev_gray, gray)))

                _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
                goal_w    = RESIZE_WIDTH // 4
                field_top = new_h // 3
                density_scores.append(
                    np.sum(thresh[field_top:, :goal_w]  > 0) +
                    np.sum(thresh[field_top:, -goal_w:] > 0)
                )
                prev_gray = gray

            cap.release()
            self.log("  ✓ 映像解析完了")

            # ── 3/6 スコアリング ──
            self.step_signal.emit(3)
            self.log("【3/6】ハイライト候補抽出中...")
            self.progress_signal.emit(base_progress + 14,
                                      f"[{video_index+1}/{total_videos}] スコアリング中")

            motion_scores  = np.array(motion_scores,  dtype=np.float32)
            density_scores = np.array(density_scores, dtype=np.float32)
            min_len = min(len(audio_scores), len(motion_scores), len(density_scores))
            if min_len == 0:
                self.log("  ✗ データ不足 → スキップ")
                continue

            audio_scores   = audio_scores[:min_len]
            motion_scores  = motion_scores[:min_len]
            density_scores = density_scores[:min_len]

            def normalize(arr):
                m = np.max(arr)
                return arr / m if m > 0 else arr

            final_scores = (
                normalize(audio_scores)   * 0.55 +
                normalize(density_scores) * 0.30 +
                normalize(motion_scores)  * 0.15
            )

            smooth = []
            for i in range(len(final_scores)):
                s = max(0, i - 5)
                e = min(len(final_scores), i + 5)
                smooth.append(np.mean(final_scores[s:e]))
            final_scores = np.array(smooth)

            sorted_idx      = np.argsort(final_scores)[::-1]
            candidate_times = []
            for idx in sorted_idx:
                if final_scores[idx] < 0.20:
                    continue
                if any(abs(idx - ex) < MIN_SCENE_INTERVAL for ex in candidate_times):
                    continue
                candidate_times.append(idx)
                if len(candidate_times) >= 20:
                    break
            candidate_times = sorted(candidate_times)
            self.log(f"  候補シーン数: {len(candidate_times)}")

            # ── 4/6 YOLO ──
            self.step_signal.emit(4)
            self.log("【4/6】YOLO解析中...")
            self.progress_signal.emit(base_progress + 16,
                                      f"[{video_index+1}/{total_videos}] YOLO解析中")

            cap = cv2.VideoCapture(VIDEO_PATH)
            for idx, t in enumerate(candidate_times):
                if self._cancel:
                    cap.release()
                    self.finished_signal.emit(False, "キャンセルされました")
                    return

                self.progress_signal.emit(
                    base_progress + 16 + int(idx / max(len(candidate_times), 1) * 4),
                    f"[{video_index+1}/{total_videos}] YOLO [{idx+1}/{len(candidate_times)}]"
                )

                # 5フレームをまとめて取得してバッチ推論（1枚ずつより高速）
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0, t - 3) * 1000)
                frames = []
                for _ in range(5):
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)

                gk = 0
                if frames:
                    results = model(frames, verbose=False, stream=False)
                    for r in results:
                        if sum(int(b.cls[0]) == 0 for b in r.boxes) >= 6:
                            gk += 1

                all_highlights.append({
                    "video": VIDEO_PATH,
                    "time":  t,
                    "score": final_scores[t] * 0.7 + (gk / len(frames) if frames else 0) * 0.3,
                    "audio": float(audio_scores[t]),
                })

            cap.release()
            self.log("  ✓ YOLO解析完了")

        # ── TOP 選定 ──
        self.log("\n全動画 TOP 選定中...")
        self.progress_signal.emit(82, "TOP選定中...")

        all_highlights.sort(key=lambda x: x["score"], reverse=True)
        final_highlights = all_highlights[:TOP_HIGHLIGHTS]
        final_highlights.sort(
            key=lambda x: (os.path.basename(x["video"]).lower(), x["time"])
        )
        self.log(f"最終ハイライト数: {len(final_highlights)}")

        if not final_highlights:
            self.finished_signal.emit(False, "ハイライト候補が見つかりませんでした")
            return

        # ── 5/6 切り出し ──
        self.step_signal.emit(5)
        self.log("\n【5/6】動画切り出し中...")
        generated = []

        for i, item in enumerate(final_highlights):
            if self._cancel:
                self.finished_signal.emit(False, "キャンセルされました")
                return

            self.progress_signal.emit(
                82 + int(i / max(len(final_highlights), 1) * 10),
                f"切り出し [{i+1}/{len(final_highlights)}]"
            )

            al         = item["audio"]
            multiplier = (2.0 if al >= 0.90 else
                          1.5 if al >= 0.75 else
                          1.2 if al >= 0.60 else 1.0)
            start = max(0, item["time"] - HIGHLIGHT_BEFORE * multiplier)
            end   = item["time"] + HIGHLIGHT_AFTER  * multiplier
            out   = os.path.join(OUTPUT_DIR, f"_hl_{i+1}.mp4")
            generated.append(out)

            subprocess.run(
                [FFMPEG, "-y",
                 "-ss", str(start), "-to", str(end),
                 "-i", item["video"],
                 "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                 "-c:a", "aac", "-ar", "44100", out],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                startupinfo=_SI
            )
            self.log(f"  [{i+1}] 完了")

        # ── 6/6 結合 ──
        self.step_signal.emit(6)
        self.log("\n【6/6】ハイライト結合中...")
        self.progress_signal.emit(93, "動画結合中...")

        list_file    = os.path.join(OUTPUT_DIR, "_hl_list.txt")
        final_output = os.path.join(OUTPUT_DIR, FINAL_HIGHLIGHT_NAME)

        with open(list_file, "w", encoding="utf-8") as f:
            for g in generated:
                f.write(f"file '{g}'\n")

        subprocess.run(
            [FFMPEG, "-y",
             "-f", "concat", "-safe", "0", "-i", list_file,
             "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-c:a", "aac", "-ar", "44100", final_output],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            startupinfo=_SI
        )

        # 後処理
        self.progress_signal.emit(98, "後処理中...")
        for g in generated:
            if os.path.exists(g):
                os.remove(g)
        if os.path.exists(list_file):
            os.remove(list_file)

        self.progress_signal.emit(100, "完了！")
        self.log(f"\n✅ 完了！\n出力: {final_output}")
        self.finished_signal.emit(True, final_output)


# ==========================================
# ドラッグ＆ドロップ対応ファイルリスト
# ==========================================

class DropListWidget(QListWidget):

    files_added = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        # NoDragDrop にするとウィジェット自体のD&Dが無効になるので DropOnly に変更
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.viewport().setAcceptDrops(True)
        self._paths: list[str] = []

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e: QDropEvent):
        added = []
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(".mp4") and path not in self._paths:
                self._paths.append(path)
                added.append(path)
                item = QListWidgetItem(f"🎬  {os.path.basename(path)}")
                item.setToolTip(path)
                self.addItem(item)
        if added:
            self.files_added.emit(added)
        e.acceptProposedAction()

    def add_files(self, paths: list[str]):
        added = []
        for path in paths:
            if path.lower().endswith(".mp4") and path not in self._paths:
                self._paths.append(path)
                added.append(path)
                item = QListWidgetItem(f"🎬  {os.path.basename(path)}")
                item.setToolTip(path)
                self.addItem(item)
        if added:
            self.files_added.emit(added)

    def remove_selected(self):
        for item in self.selectedItems():
            path = item.toolTip()
            if path in self._paths:
                self._paths.remove(path)
            self.takeItem(self.row(item))

    def clear_all(self):
        self._paths.clear()
        self.clear()

    def get_paths(self) -> list[str]:
        return list(self._paths)


# ==========================================
# スタイルシート
# ==========================================

STYLE = """
QMainWindow, QWidget {
    background-color: #0f1117;
    color: #e0e0e0;
    font-family: "Yu Gothic UI", "Meiryo UI", sans-serif;
}
QGroupBox {
    border: 1px solid #2a2d3a; border-radius: 8px;
    margin-top: 10px; padding: 8px;
    font-weight: bold; color: #7ecfff; font-size: 12px;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }

QLabel { color: #b0b8c8; font-size: 12px; }

QLineEdit {
    background: #1a1d26; border: 1px solid #2a2d3a;
    border-radius: 5px; color: #e0e0e0; padding: 5px 8px; font-size: 12px;
}
QLineEdit:focus { border-color: #7ecfff; }

QSpinBox, QDoubleSpinBox {
    background: #1a1d26; border: 1px solid #2a2d3a;
    border-radius: 5px; color: #e0e0e0;
    padding: 4px 8px; font-size: 12px; min-width: 70px;
}
QSpinBox:focus, QDoubleSpinBox:focus { border-color: #7ecfff; }
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background: #2a2d3a; border: none; width: 16px;
}

QListWidget {
    background: #090c12; border: 2px dashed #2a3a50;
    border-radius: 6px; color: #d0e8ff; font-size: 12px;
}
QListWidget::item:selected { background: #1a3a5a; border-radius: 3px; }
QListWidget::item:hover    { background: #151d2a; }

QPushButton#browse {
    background: #1e2130; border: 1px solid #3a3d4a;
    border-radius: 5px; color: #90b8e0; padding: 5px 14px; font-size: 12px;
}
QPushButton#browse:hover { background: #252838; }

QPushButton#danger {
    background: #2a1a1a; border: 1px solid #5a2a2a;
    border-radius: 5px; color: #e08080; padding: 5px 14px; font-size: 12px;
}
QPushButton#danger:hover { background: #3a2020; }

QPushButton#start {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #1a7acc, stop:1 #0f5faa);
    border: none; border-radius: 8px; color: white;
    padding: 10px 32px; font-size: 14px; font-weight: bold; min-width: 130px;
}
QPushButton#start:hover {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #2090e0, stop:1 #1570cc);
}
QPushButton#start:disabled { background: #2a2d3a; color: #555; }

QPushButton#cancel {
    background: #2a1a1a; border: 1px solid #5a2a2a;
    border-radius: 8px; color: #e08080; padding: 10px 24px;
    font-size: 13px; min-width: 100px;
}
QPushButton#cancel:hover { background: #3a2020; }
QPushButton#cancel:disabled { color: #444; border-color: #333; background: #1a1a1a; }

QPushButton#help {
    background: #1e2130; border: 1px solid #3a3d4a;
    border-radius: 12px; color: #7ecfff;
    font-size: 13px; font-weight: bold;
}
QPushButton#help:hover { background: #252838; }

QPushButton#quit {
    background: #2a1a1a; border: 1px solid #5a2a2a;
    border-radius: 8px; color: #e08080;
    padding: 10px 24px; font-size: 13px; min-width: 100px;
}
QPushButton#quit:hover { background: #3a2020; }

QProgressBar {
    background: #1a1d26; border: 1px solid #2a2d3a;
    border-radius: 6px; height: 16px;
    text-align: center; color: #e0e0e0; font-size: 11px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #1a7acc, stop:1 #00d4aa);
    border-radius: 5px;
}
QTextEdit {
    background: #090c12; border: 1px solid #1e2230;
    border-radius: 6px; color: #b8ffb8;
    font-family: "Consolas","Courier New",monospace; font-size: 11px;
}
"""

# ==========================================
# ステップ表示バー
# ==========================================

STEP_LABELS = ["音声解析", "映像解析", "スコアリング", "YOLO解析", "切り出し", "結合"]


class StepIndicator(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.labels = []
        for i, name in enumerate(STEP_LABELS):
            lbl = QLabel(f"{i+1}. {name}")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFixedHeight(28)
            lbl.setStyleSheet(
                "font-size:11px; color:#555; background:#151820;"
                "border-radius:4px; padding:2px;"
            )
            layout.addWidget(lbl, 1)
            self.labels.append(lbl)

    def set_step(self, step: int):
        for i, lbl in enumerate(self.labels):
            idx = i + 1
            if idx < step:
                lbl.setStyleSheet(
                    "font-size:11px; color:#00aa77; background:#0d1a14;"
                    "border-radius:4px; padding:2px;"
                )
            elif idx == step:
                lbl.setStyleSheet(
                    "font-size:11px; color:#fff; background:#1a5099;"
                    "border-radius:4px; padding:2px; font-weight:bold;"
                )
            else:
                lbl.setStyleSheet(
                    "font-size:11px; color:#555; background:#151820;"
                    "border-radius:4px; padding:2px;"
                )

    def reset(self):
        self.set_step(0)


# ==========================================
# メインウィンドウ
# ==========================================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("⚽ サッカーハイライト生成 AI")
        self.setMinimumSize(840, 720)
        self.resize(940, 800)
        self.setAcceptDrops(True)
        self.worker  = None
        self.thread_ = None
        self._build_ui()
        self.setStyleSheet(STYLE)
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        if os.path.isfile(FFMPEG):
            self._log(f"✅ ffmpeg 検出: {FFMPEG}")
        else:
            try:
                subprocess.run(["ffmpeg", "-version"],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL,
                               startupinfo=_SI, check=True)
                self._log("✅ ffmpeg: PATH から検出")
            except Exception:
                self._log("⚠️  ffmpeg が見つかりません！"
                          " EXE と同じフォルダに ffmpeg.exe を置いてください。")

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(14, 14, 14, 14)

        # タイトル
        title = QLabel("⚽  Soccer Highlight Generator  AI")
        title.setFont(QFont("Yu Gothic UI", 16, QFont.Weight.Bold))
        title.setStyleSheet("color:#7ecfff; margin-bottom:4px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#2a2d3a;")
        root.addWidget(sep)

        self.step_indicator = StepIndicator()
        root.addWidget(self.step_indicator)

        # ─── 動画ファイル ───
        file_group = QGroupBox("🎬 動画ファイル（ドラッグ＆ドロップ または ファイル選択・複数可）")
        fl = QVBoxLayout(file_group)
        fl.setSpacing(6)

        hint = QLabel("↓ ここに MP4 ファイルをドロップ")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet(
            "color:#3a5a78; font-size:12px;"
            "padding:4px; border-radius:4px;"
        )
        fl.addWidget(hint)

        self.file_list = DropListWidget()
        self.file_list.setMinimumHeight(130)
        self.file_list.files_added.connect(self._update_file_count)
        fl.addWidget(self.file_list)

        file_btn_row = QHBoxLayout()
        btn_add = QPushButton("＋ ファイルを追加", objectName="browse")
        btn_add.clicked.connect(self._add_files_dialog)
        file_btn_row.addWidget(btn_add)

        btn_remove = QPushButton("選択を削除", objectName="danger")
        btn_remove.clicked.connect(self._remove_selected)
        file_btn_row.addWidget(btn_remove)

        btn_clear_files = QPushButton("すべてクリア", objectName="danger")
        btn_clear_files.clicked.connect(self._clear_files)
        file_btn_row.addWidget(btn_clear_files)

        file_btn_row.addStretch()

        self.file_count_label = QLabel("0 ファイル")
        self.file_count_label.setStyleSheet("color:#7ecfff; font-size:12px; font-weight:bold;")
        file_btn_row.addWidget(self.file_count_label)

        fl.addLayout(file_btn_row)
        root.addWidget(file_group)

        # ─── 出力フォルダ ───
        out_group = QGroupBox("📁 出力フォルダ")
        og = QHBoxLayout(out_group)
        og.setSpacing(8)
        self.output_edit = QLineEdit(os.path.join(_base_dir(), "output"))
        og.addWidget(self.output_edit)
        btn_out = QPushButton("参照", objectName="browse")
        btn_out.clicked.connect(self._browse_output)
        og.addWidget(btn_out)
        root.addWidget(out_group)

        # ─── パラメータ ───
        param_group = QGroupBox("⚙️ パラメータ設定")
        pg = QGridLayout(param_group)
        pg.setSpacing(8)
        pg.setColumnStretch(1, 1)
        pg.setColumnStretch(3, 1)

        pg.addWidget(QLabel("ハイライト数"), 0, 0)
        self.sp_highlights = QSpinBox()
        self.sp_highlights.setRange(1, 100)
        self.sp_highlights.setValue(20)
        self.sp_highlights.setSuffix(" シーン")
        pg.addWidget(self.sp_highlights, 0, 1)

        pg.addWidget(QLabel("解析FPS"), 0, 2)
        self.sp_fps = QDoubleSpinBox()
        self.sp_fps.setRange(0.1, 10.0)
        self.sp_fps.setValue(1.0)
        self.sp_fps.setSingleStep(0.5)
        self.sp_fps.setSuffix(" fps")
        pg.addWidget(self.sp_fps, 0, 3)

        pg.addWidget(QLabel("前（秒）"), 1, 0)
        self.sp_before = QSpinBox()
        self.sp_before.setRange(1, 120)
        self.sp_before.setValue(12)
        self.sp_before.setSuffix(" 秒")
        pg.addWidget(self.sp_before, 1, 1)

        pg.addWidget(QLabel("後（秒）"), 1, 2)
        self.sp_after = QSpinBox()
        self.sp_after.setRange(1, 120)
        self.sp_after.setValue(8)
        self.sp_after.setSuffix(" 秒")
        pg.addWidget(self.sp_after, 1, 3)

        # ヘルプボタン
        btn_help = QPushButton("?", objectName="help")
        btn_help.setFixedSize(24, 24)
        btn_help.setToolTip("パラメータの説明")
        btn_help.clicked.connect(self._show_help)
        pg.addWidget(btn_help, 1, 4)

        root.addWidget(param_group)

        # ─── 進捗 ───
        prog_group = QGroupBox("📊 進捗")
        pl = QVBoxLayout(prog_group)
        pl.setSpacing(6)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        pl.addWidget(self.progress_bar)
        self.progress_label = QLabel("待機中...")
        self.progress_label.setStyleSheet("font-size:11px; color:#888;")
        pl.addWidget(self.progress_label)
        root.addWidget(prog_group)

        # ─── ログ ───
        log_group = QGroupBox("📋 ログ")
        ll = QVBoxLayout(log_group)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMinimumHeight(120)
        ll.addWidget(self.log_area)
        root.addWidget(log_group, 1)

        # ─── ボタン行 ───
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        btn_clr = QPushButton("ログクリア", objectName="browse")
        btn_clr.clicked.connect(self.log_area.clear)
        btn_row.addWidget(btn_clr)

        btn_row.addSpacing(12)

        self.btn_cancel = QPushButton("キャンセル", objectName="cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        btn_row.addWidget(self.btn_cancel)

        self.btn_start = QPushButton("▶  開始", objectName="start")
        self.btn_start.clicked.connect(self._start)
        btn_row.addWidget(self.btn_start)

        self.btn_quit = QPushButton("✕  終了", objectName="quit")
        self.btn_quit.clicked.connect(QApplication.quit)
        btn_row.addWidget(self.btn_quit)

        root.addLayout(btn_row)

    # ── ファイル操作ヘルパー ──
    def _update_file_count(self, _=None):
        n = len(self.file_list.get_paths())
        self.file_count_label.setText(f"{n} ファイル")

    def _add_files_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "MP4ファイルを選択", "",
            "動画ファイル (*.mp4);;すべてのファイル (*)"
        )
        if paths:
            self.file_list.add_files(paths)
            self._update_file_count()

    def _remove_selected(self):
        self.file_list.remove_selected()
        self._update_file_count()

    def _clear_files(self):
        self.file_list.clear_all()
        self._update_file_count()

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(
            self, "出力フォルダを選択", self.output_edit.text()
        )
        if d:
            self.output_edit.setText(d)

    # ── 処理制御 ──
    def _start(self):
        video_paths = self.file_list.get_paths()
        if not video_paths:
            self._log("❌ 動画ファイルが追加されていません")
            return

        params = {
            "video_paths":      video_paths,
            "output_dir":       self.output_edit.text().strip() or "output",
            "analyze_fps":      self.sp_fps.value(),
            "top_highlights":   self.sp_highlights.value(),
            "highlight_before": self.sp_before.value(),
            "highlight_after":  self.sp_after.value(),
        }

        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setValue(0)
        self.step_indicator.reset()
        self.log_area.clear()

        self.worker  = HighlightWorker(params)
        self.thread_ = QThread()
        self.worker.moveToThread(self.thread_)

        self.thread_.started.connect(self.worker.run)
        self.worker.log_signal.connect(self._log)
        self.worker.progress_signal.connect(self._update_progress)
        self.worker.step_signal.connect(self.step_indicator.set_step)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.finished_signal.connect(self.thread_.quit)
        self.thread_.finished.connect(self.thread_.deleteLater)

        self.thread_.start()

    def _cancel(self):
        if self.worker:
            self.worker.cancel()
            self._log("⚠ キャンセルリクエスト送信...")
        self.btn_cancel.setEnabled(False)

    def _show_help(self):
        from PyQt6.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setWindowTitle("パラメータの説明")
        msg.setText(
            "<table cellspacing='8'>"
            "<tr><th align='left'>項目</th><th align='left'>説明</th></tr>"
            "<tr><td><b>ハイライト数</b></td><td>最終動画に含める最大シーン数</td></tr>"
            "<tr><td><b>解析FPS</b></td><td>映像解析の間引き（低いほど高速・精度低）</td></tr>"
            "<tr><td><b>前（秒）</b></td><td>ハイライト点から何秒前を含めるか</td></tr>"
            "<tr><td><b>後（秒）</b></td><td>ハイライト点から何秒後を含めるか</td></tr>"
            "</table>"
        )
        msg.setStyleSheet(
            "QMessageBox { background-color: #0f1117; color: #e0e0e0; }"
            "QLabel { color: #e0e0e0; font-size: 12px; }"
            "QPushButton { background: #1e2130; border: 1px solid #3a3d4a;"
            "border-radius: 5px; color: #90b8e0; padding: 5px 14px; }"
        )
        msg.exec()

    def _log(self, msg: str):
        self.log_area.append(msg)
        self.log_area.moveCursor(QTextCursor.MoveOperation.End)

    def _update_progress(self, pct: int, label: str):
        self.progress_bar.setValue(pct)
        self.progress_label.setText(label)

    def _on_finished(self, success: bool, msg: str):
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        if success:
            self.step_indicator.set_step(7)
            self._log(f"\n🎉 ハイライト動画生成完了！\n→ {msg}")
        else:
            self._log(f"\n❌ 処理終了: {msg}")
        self.progress_label.setText("完了" if success else "停止")


# ==========================================
# エントリーポイント
# ==========================================


# ==========================================
# スプラッシュスクリーン
# ==========================================

class SplashScreen(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.SplashScreen |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setFixedSize(400, 200)
        self.setStyleSheet(
            "background-color: #0f1117;"
            "border: 1px solid #2a2d3a;"
            "border-radius: 12px;"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(16)

        title = QLabel("⚽  Soccer Highlight Generator")
        title.setFont(QFont("Yu Gothic UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #7ecfff;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        self.msg = QLabel("起動中...")
        self.msg.setStyleSheet("color: #b0b8c8; font-size: 12px;")
        self.msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.msg)

        bar = QProgressBar()
        bar.setRange(0, 0)
        bar.setFixedHeight(6)
        bar.setStyleSheet(
            "QProgressBar { background: #1a1d26; border: none; border-radius: 3px; }"
            "QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #1a7acc, stop:1 #00d4aa); border-radius: 3px; }"
        )
        layout.addWidget(bar)

        screen = QApplication.primaryScreen().geometry()
        self.move(
            screen.center().x() - self.width() // 2,
            screen.center().y() - self.height() // 2
        )

    def set_message(self, msg: str):
        self.msg.setText(msg)
        QApplication.processEvents()

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    splash = SplashScreen()
    splash.show()
    QApplication.processEvents()

    splash.set_message("ライブラリ読み込み中...")
    QApplication.processEvents()

    win = MainWindow()

    splash.set_message("準備完了！")
    QApplication.processEvents()

    win.show()
    splash.close()

    sys.exit(app.exec())



if __name__ == "__main__":
    main()
