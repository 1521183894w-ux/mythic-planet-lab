"""仓库约定检查（不依赖科学包，任何环境都能跑）。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UTF8_BOM = b"\xef\xbb\xbf"


def _tracked_like(pattern: str) -> list[Path]:
    skip = {"environment", ".cache", ".git", "figures"}
    return [p for p in ROOT.rglob(pattern) if not (set(p.relative_to(ROOT).parts) & skip)]


def test_powershell_scripts_have_utf8_bom():
    """PowerShell 5.1 把不带 BOM 的脚本按系统代码页（GBK）读取，中文注释会变成乱码甚至语法错误。"""
    scripts = _tracked_like("*.ps1")
    assert scripts, "没有找到任何 .ps1 文件"
    missing = [str(p.relative_to(ROOT)) for p in scripts if not p.read_bytes().startswith(UTF8_BOM)]
    assert not missing, f"以下 .ps1 文件缺少 UTF-8 BOM：{missing}"


def test_lock_file_is_explicit_with_hashes():
    """锁文件必须是 micromamba 显式格式，每个包带校验和，保证两台电脑装出完全相同的环境。"""
    lock = ROOT / "environment.win-64.lock.txt"
    assert lock.exists(), "缺少 environment.win-64.lock.txt，请运行 setup_env.ps1 -FromSpec"
    lines = lock.read_text(encoding="utf-8").splitlines()
    assert "@EXPLICIT" in lines
    urls = [ln for ln in lines if ln.startswith("https://")]
    assert len(urls) > 100
    assert all("#" in ln for ln in urls), "存在没有校验和的包条目"
