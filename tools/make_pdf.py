# -*- coding: utf-8 -*-
"""把报告 HTML 转成 PDF，方便发给别人或放进作品集。

    python tools/make_pdf.py                # 默认转 docs/report.html
    python tools/make_pdf.py --dir out      # 转生成目录里的报告

原理：调用本机已安装的 Chrome / Edge 的无头模式打印 PDF。
HTML 里的图片是 base64 内嵌的，所以生成的 PDF 是自包含的。
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def find_browser() -> str:
    for p in CANDIDATES:
        if os.path.exists(p):
            return p
    for name in ("chrome", "chromium", "msedge"):
        p = shutil.which(name)
        if p:
            return p
    return ""


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="docs", help="报告所在目录（默认 docs）")
    args = ap.parse_args()

    outdir = os.path.join(ROOT, args.dir)
    html = os.path.join(outdir, "report.html")
    if not os.path.exists(html):
        print(f"找不到 {args.dir}/report.html，请先运行：aaa-isp-lab --out {args.dir}")
        return 1
    exe = find_browser()
    if not exe:
        print("没找到 Chrome/Edge，无法生成 PDF。可以直接用浏览器打开 HTML 后 Ctrl+P。")
        return 1

    pdf = os.path.join(outdir, "report.pdf")
    url = "file:///" + html.replace("\\", "/")
    cmd = [exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
           f"--print-to-pdf={pdf}", url]
    print("调用：", os.path.basename(exe))
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(pdf):
        print(f"已生成 {pdf}（{os.path.getsize(pdf) / 1024:.0f} KB）")
        return 0
    print("生成失败")
    return 1


if __name__ == "__main__":
    sys.exit(main())
