# Instalacao do PFF

## Fluxo suportado

O fluxo suportado deste repositorio e Docker-first. Os comandos de uso diario sao executados pelos wrappers na raiz e nao dependem de `.venv` local.

Wrappers disponiveis:

```bash
./scripts/package/pff-run
./scripts/package/pff-tool-run pytest
./scripts/package/pff-tool-run ruff
./scripts/package/pff-tool-run mypy
./scripts/package/pff-tool-run pyright
./scripts/package/pff-tool-run pylint
./scripts/package/pff-tool-run black
```

## Pre-requisitos

```bash
docker --version
docker compose version
```

Requisitos minimos:

* Docker 24+
* Docker Compose 2.20+
* NVIDIA Container Toolkit apenas se voce for usar GPU

## Instalacao rapida

```bash
git clone <repo-url>
cd PFF

cp .env.example .env
cp config/infra/api_hosts.yaml.example config/infra/api_hosts.yaml
mkdir -p logs outputs

# Build inicial opcional
./scripts/package/build-images.sh all

# Validacao minima
./scripts/package/pff-run --help
./scripts/package/pff-tool-run ruff check .
```

Os wrappers tambem constroem as imagens automaticamente no primeiro uso, entao o build inicial e opcional.

## Execucao

### CLI

```bash
./scripts/package/pff-run run data/manifest.yaml
./scripts/package/pff-run generate data/manifest.txt -o data/manifest.yaml
./scripts/package/pff-run clean deep -y
```

### API

```bash
docker compose up -d --wait postgres redis api
curl http://localhost:8000/health
```

### Testes e qualidade

```bash
./scripts/package/pff-tool-run pytest -q
./scripts/package/pff-tool-run ruff check .
./scripts/package/pff-tool-run mypy src
./scripts/package/pff-tool-run pyright
```

O wrapper `./scripts/package/pff-tool-run pytest` sobe `postgres` e `redis` com `tests/.env.test` e executa os testes em um container isolado, sem depender de `outputs` do host.

## GPU

O wrapper `./scripts/package/pff-run` detecta GPU NVIDIA e seleciona `pff:cuda` quando o runtime Docker GPU estiver disponivel. Caso contrario, faz fallback explicito para `pff:cpu`.

## Limpeza

```bash
./scripts/package/pff-run clean deep -y
docker builder prune -af
docker image rm -f pff:cpu pff:cuda pff:tools
```

## Nota para mantenedores

Poetry e `.venv` local ficam restritos a manutencao avancada do projeto. Instalacao, execucao e validacao rotineiras devem usar os wrappers Docker-first.
