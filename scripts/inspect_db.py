import asyncio

from pff.infrastructure.persistence.db.connection import get_connection_pool


async def inspect_optuna():
    pool = await get_connection_pool()
    async with pool.acquire() as conn:
        studies = await conn.fetch("SELECT study_id, study_name FROM studies")
        print(f"--- Studies in DB ({len(studies)}) ---")
        for s in studies:
            trial_count = await conn.fetchval(
                "SELECT COUNT(*) FROM trials WHERE study_id = $1", s["study_id"]
            )
            print(f"ID: {s['study_id']} | Name: {s['study_name']} | Trials: {trial_count}")

        running = await conn.fetch("""
            SELECT t.trial_id, t.number, s.study_name, t.state 
            FROM trials t 
            JOIN studies s ON t.study_id = s.study_id 
            WHERE t.state = 'RUNNING'
        """)
        print(f"\n--- Running Trials ({len(running)}) ---")
        for r in running:
            print(f"Study: {r['study_name']} | Trial Num: {r['number']} | State: {r['state']}")


if __name__ == "__main__":
    asyncio.run(inspect_optuna())
