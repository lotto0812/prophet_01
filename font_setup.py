"""
日本語フォント設定用ユーティリティ
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os

def setup_japanese_font():
    """
    日本語フォントを設定する
    """
    system = platform.system()
    
    if system == "Windows":
        # Windows用の日本語フォント
        font_candidates = [
            'Yu Gothic',
            'Meiryo', 
            'MS Gothic',
            'Hiragino Sans',
            'Noto Sans CJK JP',
            'Takao',
            'IPAexGothic',
            'IPAPGothic'
        ]
    elif system == "Darwin":  # macOS
        # macOS用の日本語フォント
        font_candidates = [
            'Hiragino Sans',
            'Hiragino Kaku Gothic ProN',
            'Yu Gothic',
            'Noto Sans CJK JP'
        ]
    else:  # Linux
        # Linux用の日本語フォント
        font_candidates = [
            'Noto Sans CJK JP',
            'Takao',
            'IPAexGothic',
            'IPAPGothic',
            'VL PGothic'
        ]
    
    # 利用可能なフォントを確認
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 日本語フォントを探す
    japanese_font = None
    for font in font_candidates:
        if font in available_fonts:
            japanese_font = font
            break
    
    if japanese_font:
        plt.rcParams['font.family'] = japanese_font
        print(f"日本語フォントを設定しました: {japanese_font}")
    else:
        # フォールバック: デフォルトフォントを使用
        plt.rcParams['font.family'] = 'DejaVu Sans'
        print("日本語フォントが見つかりませんでした。デフォルトフォントを使用します。")
    
    # マイナス記号の表示設定
    plt.rcParams['axes.unicode_minus'] = False
    
    return japanese_font is not None

def test_japanese_display():
    """
    日本語表示のテスト
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, '日本語テスト\nカテゴリ別売上予測', 
            ha='center', va='center', fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('日本語フォントテスト')
    plt.show()

if __name__ == "__main__":
    setup_japanese_font()
    test_japanese_display() 