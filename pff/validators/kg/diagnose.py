from pathlib import Path

import numpy as np
import polars as pl
from pff.utils import FileManager
fm = FileManager()

print("=== DIAGNÓSTICO ESPECÍFICO DO BLOCKER AnyBURL/Ray ===\n")


def diagnosticar_entity_map(path: Path):
    print("1️⃣  Verificando entity_map em", path)
    if not path.exists():
        print("   ⚠️  entity_map.parquet não encontrado, pulei esta etapa.\n")
        return
    df = fm.read(path)
    mask = pl.col("label").str.contains("1970-01-01") | pl.col("label").str.contains(
        "9999-12-31"
    )
    bad = df.filter(mask)
    if bad.height > 0:
        print(bad)
        ids = bad["id"].to_list()
        print(f"   🆔 IDs problemáticos: {ids}\n")
        # ALTERAÇÃO: para remover, descomente:
        # clean_df = df.filter(~mask)
        # clean_df.write_parquet(path.with_suffix('.clean.parquet'))
    else:
        print("   ✅ Nenhum timestamp inválido encontrado.\n")


def diagnosticar_npy(split: str, base: Path):
    path = base / f"{split}.npy"
    print(f"2️⃣  Verificando {split}.npy em {path}")
    if not path.exists():
        print("   ⚠️  Arquivo não encontrado, pulei esta etapa.\n")
        return
    arr = np.load(path)
    print(f"   ✅ Shape: {arr.shape}\n")


def diagnosticar_regras(path: Path):
    print("3️⃣  Verificando rules_anyburl.tsv em", path)
    if not path.exists():
        print("   ⚠️  Arquivo não encontrado, pulei esta etapa.\n")
        return
    from pff.utils import FileManager

    found = False
    content = FileManager().read_text(path)
    for idx, line in enumerate(content.splitlines(), 1):
        if "1970-01-01" in line or "9999-12-31" in line:
            if not found:
                print("   ❌ Linhas com timestamps inválidos:")
                found = True
                print(f"     Linha {idx}: {line.strip()[:100]}...")
    if not found:
        print("   ✅ Nenhuma regra com timestamp inválido.\n")


def check_anyburl_rules(path: Path):
    print("4️⃣  Checando literal específico em rules_anyburl.tsv", path)
    if not path.exists():
        print("   ⚠️  Arquivo não encontrado.\n")
        return
    content = path.read_text(encoding="utf-8")
    literal = "1970-01-01T22:59:59.151-03:00"
    if literal in content:
        print(f"   ❌ Literal '{literal}' encontrado nas regras!")
        for idx, line in enumerate(content.splitlines(), 1):
            if literal in line:
                print(f"     Linha {idx}: {line.strip()}")
    else:
        print("   ✅ Literal não encontrado.\n")


def diagnosticar_entidades_orfas(base: Path):
    print("5️⃣  Verificação de Entidades Órfãs")
    mp = base / "entity_map.parquet"
    print("   🔍 Carregando mapeamento de entidades:", mp)
    if not mp.exists():
        print("   ⚠️  entity_map.parquet não encontrado.\n")
        return
    em = fm.read(mp)
    print(f"   📊 Total de entidades: {em.height:,}")

    sets = {}
    for split in ["train", "valid", "test"]:
        npy = base / f"{split}.npy"
        print(f"   • {split}.npy:", npy)
        if not npy.exists():
            print("     ⚠️  Arquivo não encontrado")
            sets[split] = set()
            continue
        arr = fm.read(npy)
        ents = set(np.unique(arr[:, [0, 2]].flatten()))
        sets[split] = ents
        print(f"     🎯 Entidades únicas: {len(ents):,}")
    orphans = {s: sets[s] - sets.get("train", set()) for s in ["valid", "test"]}
    print("\n   ❓ Entidades órfãs (não no treino):")
    for s in orphans:
        print(f"     Em {s}: {len(orphans[s]):,}")
    if orphans.get("test"):
        print("   📝 Exemplos de órfãs no teste:")
        for eid in list(orphans["test"])[:5]:
            lbl = em.filter(pl.col("id") == eid)["label"][0]
            print(f"     ID {eid}: {lbl}")


def find_problematic_entity():
    from pff.utils import FileManager

    base = Path("outputs/pyclause")
    print("\n🔍 Buscando a entidade problemática específica...")
    mp = base / "entity_map.parquet"
    df = FileManager().read_parquet(mp)
    target = "2022-09-19T18:56:18.000-03:00"
    sel = df.filter(pl.col("label") == target)
    if sel.height == 0:
        print(f"   ❌ '{target}' não encontrada no entity_map")
        return None
    eid = sel["id"][0]
    print(f"   ✅ Encontrada ID {eid} para label '{target}'")

    info = {}
    for split in ["train", "valid", "test"]:
        npy = base / f"{split}.npy"
        if npy.exists():
            data = fm.read(npy)
            present = eid in set(data[:, [0, 2]].flatten())
            info[split] = present
            print(f"     {split}: {'✅' if present else '❌'}")
    return eid, target, info


def analyze_orphan_pattern():
    """
    Analyzes the pattern of "orphan" entities in the test set that do not appear in the training set.

    This function loads entity mappings and train/test splits, identifies orphan entity IDs present in the test set but absent from the training set, and prints statistics and examples of these orphans. It also detects and reports patterns among the orphan labels, such as timestamps and unusually long IDs.

    Outputs:
        - Total number of orphan entities.
        - Example labels of orphan entities.
        - Detected patterns among orphan labels (timestamps and long IDs).
    """
    base = Path("outputs/pyclause")
    print("\n📊 Analisando padrão de órfãs...")
    em = fm.read(base / "entity_map.parquet")
    train = fm.read(base / "train.npy")[:, [0, 2]].flatten()
    test = fm.read(base / "test.npy")[:, [0, 2]].flatten()
    train_set, test_set = set(train), set(test)
    orphans = test_set - train_set
    print(f"   📈 Total de órfãs: {len(orphans):,}")
    labels = [em.filter(pl.col("id") == eid)["label"][0] for eid in list(orphans)[:100]]
    print("\n🏷️ Exemplos de labels órfãs:")
    for i, label in enumerate(labels[:10], 1):
        print(f"     {i:2d}. {label}")
    ts = [label for label in labels if "T" in label and ":" in label]
    uid = [label for label in labels if len(label) > 20]
    print("\n   🧩 Padrões detectados:")
    print(f"     Timestamps: {len(ts)} ({len(ts) / len(labels) * 100:.1f}%)")
    print(f"     IDs longos: {len(uid)} ({len(uid) / len(labels) * 100:.1f}%)")


if __name__ == "__main__":
    base = Path("outputs/pyclause")
    # Gerais
    diagnosticar_entity_map(base / "entity_map.parquet")
    for sp in ["train", "valid", "test"]:
        diagnosticar_npy(sp, base)
    diagnosticar_regras(base / "rules_anyburl.tsv")
    check_anyburl_rules(base / "rules_anyburl.tsv")
    diagnosticar_entidades_orfas(base)
    # Blocker-specific
    result = find_problematic_entity()
    if result:
        eid, ts, info = result
        print(
            f"\n📋 Resumo da entidade problemática: ID={eid}, ts={ts}, splits={info}\n"
        )
        if not info.get("train") and info.get("test"):
            print("❌ CONFIRMADO: Entidade órfã causando o erro!\n")
    analyze_orphan_pattern()
