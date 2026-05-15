# Matriz Docker do PFF

Data: 2026-05-13

Este documento registra os perfis suportados para empacotamento Docker-first do PFF, os
smokes mínimos e as fontes usadas para orientar a estratégia de build/runtime.

## Fontes

| Tema | Fonte | Decisão aplicada |
|---|---|---|
| Multi-stage build | Docker Docs: <https://docs.docker.com/build/building/multi-stage/> | Separar `builder`, `runtime-base`, `runtime-cpu`, `runtime-cuda`, `tools` e `test`. |
| Cache e contexto | Docker Docs: <https://docs.docker.com/build/cache/optimize/> | Manter `.dockerignore`, limpar caches de instalação e evitar copiar artefatos grandes para o contexto. |
| BuildKit | Docker Docs: <https://docs.docker.com/build/buildkit/> | Detectar BuildKit/buildx quando disponível, mas manter fallback para builder legado. |
| Cache GitHub Actions | Docker Docs: <https://docs.docker.com/build/cache/backends/gha/> | Usar `cache-from/cache-to type=gha` via `docker/build-push-action` para cache estável no CI. |
| GPU em Windows/WSL2 | Docker Docs: <https://docs.docker.com/desktop/features/gpu/> | Suportar GPU NVIDIA em Windows via Docker Desktop com backend WSL2. |
| WSL2 backend | Docker Docs: <https://docs.docker.com/docker-for-windows/wsl/> | Documentar WSL2 como caminho viável para Windows e atenção a memória/disco do VM backend. |
| NVIDIA Container Toolkit | NVIDIA Docs: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html> | Exigir driver NVIDIA no host e runtime Docker com suporte a `--gpus all` para `pff:cuda`. |
| PyTorch wheels | PyTorch Docs: <https://pytorch.org/get-started/locally/> | Usar lock CPU por padrão e instalar o wheel CUDA apenas no target `cuda`. |
| Poetry groups | Poetry Docs: <https://python-poetry.org/docs/managing-dependencies/> | Não modelar CPU/CUDA como grupos opcionais conflitantes porque o resolvedor avalia grupos juntos. |

## Perfis

| Perfil | Build | Runtime esperado | Smoke mínimo | Status |
|---|---|---|---|---|
| Linux CPU-only | `./scripts/package/build-images.sh cpu` | `pff:cpu`, `torch==2.7.0+cpu`, `torch.version.cuda is None` | `./scripts/package/pff-run --help`; `./scripts/package/pff-run hpo --trials 1 --synthetic-data --no-dashboard --no-update-config --no-bert` | Suportado |
| Linux NVIDIA/CUDA | `./scripts/package/build-images.sh cuda` | `pff:cuda`, CUDA exposta por `--gpus all` | `./scripts/package/smoke-package.sh` em host NVIDIA | Suportado quando o host tem driver e NVIDIA Container Toolkit |
| Linux com GPU integrada/sem NVIDIA | `./scripts/package/build-images.sh cpu` | Fallback CPU explícito | `./scripts/package/pff-run --help`; `./scripts/package/pff-tool-run pytest -q <teste relevante>` | Suportado como CPU |
| Windows/WSL2 NVIDIA | Build dentro do WSL2 ou Docker Desktop | `pff:cuda` se Docker Desktop expuser GPU; caso contrário `pff:cpu` | `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`; depois `./scripts/package/smoke-package.sh` | Viável, dependente do backend WSL2/GPU-PV |
| macOS | `./scripts/package/build-images.sh cpu` | CPU; sem CUDA NVIDIA | `./scripts/package/pff-run --help` e testes unitários relevantes | Viável para desenvolvimento CPU; não é alvo CUDA |
| CI | `./scripts/package/build-images.sh cpu` ou target específico | CPU por padrão; CUDA apenas em runner GPU | `./scripts/package/measure-image-sizes.sh`; testes estáticos Docker; smoke CPU quando Docker daemon estiver disponível | Recomendado |

## Medição de tamanho

Use:

```bash
./scripts/package/measure-image-sizes.sh
./scripts/package/measure-image-sizes.sh --baseline outputs/docker-image-sizes-baseline.tsv
```

O formato TSV é estável:

```text
image	status	bytes	gib	baseline_bytes	delta_gib	delta_pct	budget_gib	budget_status
```

Orçamentos padrão:

| Imagem | Orçamento |
|---|---:|
| `pff:cpu` | 3 GiB |
| `pff:cuda` | 10 GiB |
| `pff:tools` | 15 GiB |
| `pff:test` | 16.5 GiB |

As variáveis `PFF_IMAGE_BUDGET_CPU_GB`, `PFF_IMAGE_BUDGET_CUDA_GB`,
`PFF_IMAGE_BUDGET_TOOLS_GB`, `PFF_IMAGE_BUDGET_TEST_GB` e
`PFF_IMAGE_BUDGET_DEFAULT_GB` ajustam esses limites sem alterar o script.
Use `--fail-on-budget` para transformar estouro de orçamento em falha de CI.

## Decisões

1. O wrapper `./scripts/package/pff-run` continua GPU-first em runtime: quando GPU NVIDIA e runtime Docker GPU estão disponíveis, ele escolhe `pff:cuda`; caso contrário, usa `pff:cpu`.
2. O build padrão continua CPU-only para evitar gerar imagens pesadas sem necessidade.
3. `tools` e `test` são targets explícitos porque carregam dependências de desenvolvimento, lint, teste e browser.
4. O lock principal resolve a wheel `torch==2.7.0+cpu` pelo índice CPU, enquanto o requisito público fica em `torch==2.7.0` para aceitar builds CPU e CUDA sem conflito de metadata.
5. O target CUDA troca explicitamente para `torch==2.7.0+cu128` e `triton==3.3.0` a partir do índice oficial CUDA 12.8.
6. BuildKit é usado quando disponível; o fluxo mantém compatibilidade com builder legado porque alguns hosts ainda não têm `docker buildx`.
7. O workflow de CI carrega `pff:ci` no Docker daemon com `load: true`, mede o tamanho com TSV, falha quando ultrapassa o orçamento e publica `outputs/benches/docker/image-sizes-ci.tsv` como artefato.
