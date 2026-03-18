import pandas as pd
from pathlib import Path


def process_ae(filepath: str) -> list[dict]:
    df = pd.read_csv(filepath)
    chunks = []

    for _, row in df.iterrows():
        if pd.isna(row.get('Org name')) or str(row.get('Org name')).strip() == '':
            continue

        type1_attendances = pd.to_numeric(row.get('A&E attendances Type 1', 0), errors='coerce') or 0
        type1_breaches = pd.to_numeric(row.get('Attendances over 4hrs Type 1', 0), errors='coerce') or 0
        waits_12hr = pd.to_numeric(row.get('Patients who have waited 12+ hrs from DTA to admission', 0), errors='coerce') or 0
        breach_rate = (type1_breaches / type1_attendances * 100) if type1_attendances > 0 else 0


        total_activity = type1_attendances + pd.to_numeric(row.get('A&E attendances Type 2', 0), errors='coerce') or 0
        if total_activity == 0:
            continue

        text = (
            f"In January 2025, {row['Org name']} (code: {row['Org Code']}) "
            f"recorded {int(type1_attendances):,} Type 1 A&E attendances. "
            f"{int(type1_breaches):,} patients waited over 4 hours "
            f"({breach_rate:.1f}% breach rate). "
            f"{int(waits_12hr):,} patients waited 12 or more hours from decision to admit. "
            f"This trust is in the {row.get('Parent Org', 'Unknown')} region."
        )

        chunks.append({
            "text": text,
            "metadata": {
                "source": "AE",
                "org_code": row['Org Code'],
                "org_name": row['Org name'],
                "period": row['Period'],
            }
        })

    return chunks

def process_rtt(filepath: str) -> list[dict]:
    df = pd.read_csv(filepath).copy()
    chunks = []

    week_cols = [c for c in df.columns if c.startswith('Gt')]
    
    group_cols = ['Provider Org Name', 'Provider Org Code', 
                  'Treatment Function Name', 'Period']
    
    df[week_cols] = df[week_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    df['Total All'] = pd.to_numeric(df['Total All'], errors='coerce').fillna(0)

    grouped = df.groupby(group_cols, as_index=False)[week_cols + ['Total All']].sum()

    for _, row in grouped.iterrows():
        total = row['Total All']
        if total == 0:
            continue

        over_18_cols = [c for c in week_cols if any(
            f'Gt {w} To' in c for w in range(18, 105)
        )]
        over_52_cols = [c for c in week_cols if any(
            f'Gt {w} To' in c for w in range(52, 105)
        )]

        waiting_over_18 = int(row[over_18_cols].sum()) if over_18_cols else 0
        waiting_over_52 = int(row[over_52_cols].sum()) if over_52_cols else 0
        pct_over_18 = (waiting_over_18 / total * 100) if total > 0 else 0

        text = (
            f"In January 2025, {row['Provider Org Name']} "
            f"(code: {row['Provider Org Code']}) had {int(total):,} patients "
            f"on incomplete RTT pathways for {row['Treatment Function Name']}. "
            f"{waiting_over_18:,} patients ({pct_over_18:.1f}%) were waiting "
            f"over 18 weeks. {waiting_over_52:,} patients were waiting over 52 weeks."
        )

        chunks.append({
            "text": text,
            "metadata": {
                "source": "RTT",
                "org_code": row['Provider Org Code'],
                "org_name": row['Provider Org Name'],
                "treatment": row['Treatment Function Name'],
                "period": row['Period'],
            }
        })

    return chunks

if __name__ == "__main__":
    ae_chunks = process_ae("data/Monthly-AE-January-2025.csv")
    print(f"A&E chunks: {len(ae_chunks)}")
    print("\nSample A&E chunk:")
    print(ae_chunks[0]['text'])

    rtt_chunks = process_rtt("data/20250131-RTT-January-2025-full-extract-revised.csv")
    print(f"\nRTT chunks: {len(rtt_chunks)}")
    print("\nSample RTT chunk:")
    print(rtt_chunks[0]['text'])
    
    import json

    all_chunks = ae_chunks + rtt_chunks
    print(f"\nTotal chunks: {len(all_chunks)}")

    with open("data/chunks.json", "w") as f:
        json.dump(all_chunks, f, indent=2)

    print("Chunks saved to data/chunks.json")