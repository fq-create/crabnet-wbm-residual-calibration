from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
EMBEDDING_DIR = ROOT / "data" / "element_properties"

MAT2VEC_PATH = EMBEDDING_DIR / "mat2vec.csv"

FUSION_SPECS = [
    ("MDS_64_cos_zscore.csv", "mat2vec+S64.csv"),
    ("MDS_32_cos_zscore.csv", "mat2vec+S32.csv"),
    ("MDS_E_64_cos_zscore.csv", "mat2vec+L64.csv"),
    ("MDS_E_32_cos_zscore.csv", "mat2vec+L32.csv"),
    ("MDS_F_32_cos_zscore.csv", "mat2vec+H32.csv"),
    ("MDS_F_64_cos_zscore.csv", "mat2vec+H64.csv"),
]


def read_embedding(path):
    df = pd.read_csv(path, dtype=str)
    element_col = df.columns[0]
    df = df.rename(columns={element_col: "element"})
    df["element"] = df["element"].astype(str)

    if df["element"].duplicated().any():
        dupes = df.loc[df["element"].duplicated(), "element"].tolist()
        raise ValueError(f"Duplicate elements found in {path}: {dupes}")

    return df


def build_concat_embedding(mat2vec, source_name, output_name):
    source_path = EMBEDDING_DIR / source_name
    output_path = EMBEDDING_DIR / output_name

    source = read_embedding(source_path)

    common_elements = set(mat2vec["element"]).intersection(source["element"])
    if not common_elements:
        raise ValueError(
            f"No common elements found between mat2vec.csv and {source_name}."
        )

    # Keep row order from the MDS table, which is the target 103-element subset.
    source_common = source[source["element"].isin(common_elements)].copy()
    mat2vec_common = (
        mat2vec[mat2vec["element"].isin(common_elements)]
        .set_index("element")
        .loc[source_common["element"]]
        .reset_index()
    )

    mat2vec_features = mat2vec_common.drop(columns=["element"]).reset_index(drop=True)
    source_features = source_common.drop(columns=["element"]).reset_index(drop=True)

    concat_features = pd.concat([mat2vec_features, source_features], axis=1)
    concat_features.columns = [str(i) for i in range(concat_features.shape[1])]

    output = pd.concat(
        [source_common[["element"]].reset_index(drop=True), concat_features],
        axis=1,
    )
    output.to_csv(output_path, index=False)

    return {
        "source": source_name,
        "output": output_name,
        "mat2vec_rows": len(mat2vec),
        "source_rows": len(source),
        "common_elements": len(common_elements),
        "output_rows": output.shape[0],
        "output_columns": output.shape[1],
        "mat2vec_features": mat2vec_features.shape[1],
        "source_features": source_features.shape[1],
    }


def main():
    mat2vec = read_embedding(MAT2VEC_PATH)

    summaries = [
        build_concat_embedding(mat2vec, source_name, output_name)
        for source_name, output_name in FUSION_SPECS
    ]

    for item in summaries:
        print(
            f"{item['source']} -> {item['output']} | "
            f"common elements: {item['common_elements']} | "
            f"output shape: {item['output_rows']} rows x "
            f"{item['output_columns']} columns | "
            f"features: {item['mat2vec_features']} + {item['source_features']}"
        )


if __name__ == "__main__":
    main()
