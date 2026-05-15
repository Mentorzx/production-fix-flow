# Stack ZIP + mmap + cache + paralelismo + JSON rapido (e caminho para Rust)

Este documento descreve, em "pensar em voz alta", o que acontece hoje no fluxo de leitura de ZIP + parse de JSON, porque isso e rapido ate certo ponto, onde aparecem os gargalos, e como encostar em Rust sem mexer no CPython (so com extensoes/bindings). Ele tambem lista fontes atuais para ZIP, zstd, GIL e PyO3/maturin.

## 1) ZIP nao e so um arquivo: tem indice no final

O formato ZIP tem um "central directory" no fim do arquivo. Para listar membros (`namelist()`), o leitor precisa abrir o arquivo e ler estruturas de indice. Isso ja cria um custo fixo quando o ZIP e aberto muitas vezes.

No FileManager, isso e amortizado com cache de membros (`_cached_zip_members`) usando assinatura `(mtime, size)`, evitando repetir o inventario quando o ZIP nao mudou.

## 2) mmap: o truque sujo do SO para reduzir syscalls

`mmap` permite tratar o arquivo como um buffer gigante de bytes e deixa o SO paginar do disco conforme necessario. Isso reduz chamadas pequenas de `read()` e aproveita o page cache do kernel.

Importante: `mmap` nao remove o custo dominante de descompressao e parse de JSON. Ele reduz custo de I/O e syscalls, mas nao elimina alocacao de objetos Python.

## 3) zipfile + ISA-L (zipfile-isal): turbina no DEFLATE

`zipfile` usa zlib/DEFLATE. A dependencia opcional `zipfile-isal` faz monkey patch no `zipfile` para usar ISA-L, acelerando a descompressao DEFLATE quando o gargalo e CPU de descompressao.

Se o gargalo maior estiver no parse de JSON e criacao de objetos Python, essa turbina ajuda, mas nao faz milagre.

## 4) Paralelismo em duas fases: onde o custo se esconde

Hoje o fluxo de `load_zip` tem duas fases:

Fase A - Leitura: processa chunks, abre um ZipFile por chunk e faz `zf.read(member)`.
Fase B - Parse: distribui os bytes e aplica `handler.load_bytes` (msgspec/orjson/Polars etc.).

Quando o backend e processos, isso pode causar trafego extra de bytes entre processos:

- bytes extraidos no worker da fase A
- bytes enviados para o processo pai
- bytes reenviados para a fase B

Isso e modular, mas paga pedagio de copia/serializacao.

## 5) Mudanca aplicada: leitura + parse fundidos no worker

Para reduzir esse pedagio, foi adicionado um caminho com `fuse_processing=True` (default quando `parallel=True`), que:

- abre o ZIP no worker
- le cada membro do chunk
- ja parseia com o handler correspondente
- devolve o resultado final para o processo pai

Isso elimina a segunda passagem de IPC dos bytes e reduz pico de memoria.

## 6) JSON rapido: msgspec e buffers reutilizaveis

O JSON ja usa msgspec com `Decoder/Encoder` reutilizaveis, com buffers thread-local para reduzir alocacoes. Isso e um bom caminho para Python competir em desempenho.

Limite: 14k JSONs ainda significam 14k objetos Python, o que sempre tera overhead nao trivial.

## 7) Por que Rust parece bruxaria nesses casos

Tres fatores aparecem sempre:
1) Formato - zstd (um stream unico) costuma ser mais amigavel que ZIP com milhares de entradas pequenas.
2) Sem GIL - Rust pode paralelizar parse e descompressao sem travas do interpretador.
3) Menos objetos dinamicos - parse tipado e layouts mais compactos evitam toneladas de dict/list.

## 8) Caminhos realistas para chegar perto de Rust sem mexer no Python

### 8.1) Extensao nativa com PyO3 + maturin

Criar um modulo nativo que:

- abre ZIP ou zstd
- descomprime e parseia
- retorna um lote (Arrow/Polars) ou objetos agregados

Isso deixa Python como orquestrador e move o hot path para Rust. PyO3 permite liberar o GIL (`Python::detach`) para executar em paralelo de verdade, e maturin simplifica build/wheels.

### 8.2) Usar mais Rust "disfarcado" via Polars/Arrow

Se o dado virar NDJSON grande, Parquet ou Arrow, o parse e o processamento pesado ficam do lado Rust (Polars). O Python vira casca.

### 8.3) Evitar copiar bytes entre processos

Mesmo em Python puro, o ganho vem de fundir "ler + parse" no mesmo worker. Isso reduz trafego e melhora escala com processos.

### 8.4) Trocar ZIP por um container mais amigavel

ZIP com milhares de entradas pequenas tem overhead alto por arquivo. Um stream unico (tar.zst, NDJSON.zst) reduz metadados e melhora throughput. Python-zstandard oferece API de streaming com cuidado de thread safety (criar um compressor por thread/worker).

## 9) GIL: o limite estrutural

Threads em CPython ainda competem no GIL para bytecode Python. Processos escalam CPU mas pagam overhead de IPC. Rust via extensao permite rodar trabalho pesado fora do GIL e reaproveitar CPU.

## 10) Mudancas recomendadas (alinhadas ao stack atual)

1) Manter `fuse_processing` como caminho padrao quando `parallel=True` para reduzir copias.
2) Manter `zipfile-isal` como aceleracao opcional para DEFLATE.
3) Para workloads extremos, avaliar conversao de datasets para NDJSON/Parquet (ou zstd streams) e usar Polars.
4) Se o ganho justificar, criar extensao Rust com PyO3 para descompressao + parse e expor interface Python compat.

## Fontes consultadas

- Python mmap: https://docs.python.org/3/library/mmap.html
- Python zlib/zipfile: https://docs.python.org/3/library/zlib.html
- zipfile-isal (PyPI): https://pypi.org/project/zipfile-isal/
- python-isal docs: https://python-isal.readthedocs.io/en/stable/
- python-zstandard docs: https://python-zstandard.readthedocs.io/en/latest/
- PyO3: https://pyo3.rs/
- maturin: https://www.maturin.rs/
- PEP 703 (GIL opcional): https://peps.python.org/pep-0703/
- PEP 784 (zstd na stdlib): https://peps.python.org/pep-0784/
- Context7: /indygreg/python-zstandard (API usage), /pyo3/pyo3 (GIL e extensoes)
