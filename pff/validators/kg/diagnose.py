from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from pff.utils import FileManager, logger

fm = FileManager()


def diagnosticar_entity_map(path: Path) -> None:
    logger.info(f"Verificando entity_map em {path}")
    if not path.exists():
        logger.info("entity_map.parquet não encontrado, pulando etapa")
        return
    df = fm.read(path)
    mask = pl.col("label").str.contains("1970-01-01") | pl.col("label").str.contains("9999-12-31")
    bad = df.filter(mask)
    if bad.height > 0:
        ids = bad["id"].to_list()
        logger.warning(f"Found labels with invalid timestamps: ids={ids}")
    else:
        logger.info("Nenhum timestamp invalido encontrado")


def diagnosticar_npy(split: str, base: Path) -> None:
    path = base / f"{split}.npy"
    logger.info(f"Verificando {split}.npy em {path}")
    if not path.exists():
        logger.info("Arquivo nao encontrado, pulando etapa")
        return
    arr = np.load(path)
    logger.debug(f"Shape: {arr.shape}")


def diagnosticar_regras(path: Path) -> None:
    logger.info(f"Verificando rules_anyburl.tsv em {path}")
    if not path.exists():
        logger.info("Arquivo não encontrado, pulando etapa")
        return
    content = fm.read_text(path)
    for idx, line in enumerate(content.splitlines(), 1):
        if "1970-01-01" in line or "9999-12-31" in line:
            logger.warning(f"Line {idx} contains invalid timestamp: {line.strip()[:100]}...")
            return
    logger.info("Nenhuma regra com timestamp invalido")


def check_anyburl_rules(path: Path) -> None:
    logger.info(f"Checando literal específico em rules_anyburl.tsv {path}")
    if not path.exists():
        logger.info("Arquivo não encontrado")
        return
    content = path.read_text(encoding="utf-8")
    literal = "1970-01-01T22:59:59.151-03:00"
    if literal in content:
        for idx, line in enumerate(content.splitlines(), 1):
            if literal in line:
                logger.warning(f"Literal '{literal}' found at line {idx}: {line.strip()}")
                break
    else:
        logger.info("Literal nao encontrado")


def diagnosticar_entidades_orfas(base: Path) -> None:
    logger.info("Verificação de Entidades Órfãs")
    mp = base / "entity_map.parquet"
    logger.info(f"Carregando mapeamento de entidades: {mp}")
    if not mp.exists():
        logger.info("entity_map.parquet não encontrado")
        return
    em = fm.read(mp)
    logger.info(f"Total de entidades: {em.height:,}")

    sets: dict[str, set[int]] = {}
    for split in ["train", "valid", "test"]:
        npy = base / f"{split}.npy"
        logger.info(f"{split}.npy: {npy}")
        if not npy.exists():
            logger.info("Arquivo não encontrado")
            sets[split] = set()
            continue
        arr = fm.read(npy)
        ents = set(np.unique(arr[:, [0, 2]].flatten()))
        sets[split] = ents
        logger.info(f"Entidades únicas: {len(ents):,}")

    orphans = {s: sets[s] - sets.get("train", set()) for s in ["valid", "test"]}
    logger.info("Entidades órfãs (não no treino):")
    for split, ids in orphans.items():
        logger.info(f"  Em {split}: {len(ids):,}")
    if orphans.get("test"):
        logger.info("Exemplos de órfãs no teste:")
        for eid in list(orphans["test"])[:5]:
            lbl = em.filter(pl.col("id") == eid)["label"][0]
            logger.info(f"  ID {eid}: {lbl}")


def find_problematic_entity() -> None:
    base = Path("outputs/pyclause")
    logger.info("Buscando entidade problemática específica")
    mp = base / "entity_map.parquet"
    df = fm.read_parquet(mp)
    target = "2022-09-19T18:56:18.000-03:00"
    sel = df.filter(pl.col("label") == target)
    if sel.height == 0:
        logger.info(f"'{target}' não encontrada no entity_map")
        return
    eid = sel["id"][0]
    logger.info(f"Encontrada ID {eid} para label '{target}'")
    for split in ["train", "valid", "test"]:
        npy = base / f"{split}.npy"
        if not npy.exists():
            continue
        arr = fm.read(npy)
        present = target in arr
        logger.info(f"{split}: {'presente' if present else 'ausente'}")


if __name__ == "__main__":
    base = Path("outputs/pyclause")
    diagnosticar_entity_map(base / "entity_map.parquet")
    diagnosticar_npy("train", base)
    diagnosticar_regras(base / "rules_anyburl.tsv")
    check_anyburl_rules(base / "rules_anyburl.tsv")
    diagnosticar_entidades_orfas(base)
    find_problematic_entity()
