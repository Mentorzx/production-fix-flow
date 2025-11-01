param(
  [string]$OutFile      = "tree.txt",
  [string]$AnnotatedOut = "tree_commented.txt"
)

$scriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Resolve-Path (Join-Path $scriptDir "..\..")
Push-Location $projectRoot

chcp 65001 > $null
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$pyCode = @'
import os, sys
sys.stdout.reconfigure(encoding='utf-8')

def tree(path, prefix=''):
    entries = sorted(
        os.scandir(path),
        key=lambda e: (e.is_file(), e.name.lower())
    )
    lines = []
    for idx, entry in enumerate(entries):
        n = entry.name
        # ignora arquivos e pastas indesejados
        if (
            n.startswith('.')
            or n in ('env', '__pycache__', 'build', 'dist', 'mlruns', 'checkpoints')
            or n.endswith('.pyc')
        ):
            continue
        connector = '└── ' if idx == len(entries) - 1 else '├── '
        line = prefix + connector + n
        if entry.is_dir():
            lines.append(line)
            new_prefix = prefix + ('    ' if idx == len(entries) - 1 else '│   ')
            lines.extend(tree(os.path.join(path, n), new_prefix))
        else:
            size = entry.stat().st_size
            if size < 1024:
                desc = f"{size} B"
            elif size < 1024**2:
                desc = f"{size / 1024:.2f} KB"
            else:
                desc = f"{size / 1024**2:.2f} MB"
            lines.append(f"{line}  # {desc}")
    return lines

if __name__ == "__main__":
    for l in tree(os.getcwd()):
        print(l)
'@

$pyCode | python -X utf8 - > $OutFile

Get-Content $OutFile -Encoding utf8

Pop-Location
