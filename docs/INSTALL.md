# 📦 Instalação do PFF

## Instalação Rápida

```bash
python install.py
```

## Plataformas Suportadas

✅ Windows 10/11 (CPU/CUDA)
✅ Linux (Ubuntu, Debian, CentOS, etc.)
✅ macOS 10.15+

## Dependências por Plataforma

### 🪟 Windows

- Inclui: pywin32, windows-curses
- Python: 3.12+
- CUDA: Detectado automaticamente

### 🐧 Linux

- Inclui: python-daemon, systemd-python
- Python: 3.12+

### 🍎 macOS

- Inclui: pyobjc-core
- Python: 3.12+

## Solução de Problemas

### Erro: "Poetry não encontrado"
```bash
pip install poetry
```

### Erro: "torch-scatter build failed"
```bash
# Windows - instale Visual Studio Build Tools
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Depois execute:
poetry run pip install torch-scatter --no-build-isolation
```

### Erro: "Dependência X não instalada"
```bash
# Windows
powershell -ExecutionPolicy Bypass -File install_special.ps1

# Linux/macOS
bash install_special.sh
```

## Build from Source

```bash
poetry build
# Arquivos em dist/
```

## Desenvolvimento

```bash
poetry install -E dev
poetry run pytest
poetry run black .
poetry run mypy .
```
