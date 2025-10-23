# from __future__ import annotations

# from pathlib import Path

# from celery import Celery

# from .miner import KGMiner

# """Celery tasks that trigger AnyBURL/PyClause training for the KG."""

# app = Celery("pff_kg")
# # Carrega configurações padrão se existir módulo; caso contrário usa defaults
# app.config_from_object("pff.config.celery", namespace="CELERY", silent=True)


# def _miner(cfg_path: str | Path = "config/kg.yaml") -> KGMiner:
#     """Helper que carrega configuração e devolve um *KGMiner*."""
#     return KGMiner.from_file(cfg_path)


# @app.task(name="kg.train_anyburl")
# def train_anyburl(cfg_path: str | Path = "config/kg.yaml") -> None:  # noqa: D401
#     _miner(cfg_path).train_anyburl()


# @app.task(name="kg.train_pyclause")
# def train_pyclause(cfg_path: str | Path = "config/kg.yaml") -> None:  # noqa: D401
#     _miner(cfg_path).train_pyclause()


# @app.task(name="kg.train_full")
# def train_full(cfg_path: str | Path = "config/kg.yaml") -> None:  # noqa: D401
#     _miner(cfg_path).full_run()
