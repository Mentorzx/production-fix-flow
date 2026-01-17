import optuna
from pff import settings


def check_trial():
    url = f"postgresql+psycopg2://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    study = optuna.load_study(study_name="pff_kg_real_dslfm_kgc", storage=url)
    trials = study.get_trials()
    if not trials:
        print("No trials found")
        return

    t = trials[0]
    for t in trials:
        if t.number >= 0:
            print(f"Trial {t.number + 1}")
            print(f"  State: {t.state}")
            print(f"  User attrs: {t.user_attrs}")


if __name__ == "__main__":
    check_trial()
