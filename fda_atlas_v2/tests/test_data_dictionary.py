import pandas as pd

from fda_atlas_v2.data_dictionary import build_data_dictionary


def test_dictionary_covers_every_field_and_marks_primary_keys_nonnullable(tmp_path):
    pd.DataFrame(
        {
            "release_id": ["r"],
            "trait_id": ["pd"],
            "display_name": ["PD"],
            "gene_set_version": ["v"],
            "n_genes": [1000],
            "extra_field": ["x"],
        }
    ).to_parquet(tmp_path / "trait_registry.parquet", index=False)
    dictionary = build_data_dictionary("r", tmp_path)
    trait_rows = dictionary[dictionary["table_name"].eq("trait_registry")]
    assert set(trait_rows["column_name"]) == {
        "release_id", "trait_id", "display_name", "gene_set_version", "n_genes",
        "extra_field",
    }
    nullable = trait_rows.set_index("column_name")["nullable"]
    assert not nullable["release_id"]
    assert not nullable["trait_id"]
    assert nullable["extra_field"]
    assert "data_dictionary" in set(dictionary["table_name"])


def test_dictionary_uses_table_specific_cross_species_availability_meaning(tmp_path):
    for table_name in ("dataset_registry", "cross_species_effects"):
        pd.DataFrame(
            {"release_id": ["r"], "availability_status": ["available"]}
        ).to_parquet(tmp_path / f"{table_name}.parquet", index=False)
    dictionary = build_data_dictionary("r", tmp_path).set_index(
        ["table_name", "column_name"]
    )
    dataset_text = dictionary.loc[
        ("dataset_registry", "availability_status"), "description"
    ]
    cross_species_text = dictionary.loc[
        ("cross_species_effects", "availability_status"), "description"
    ]
    assert "source availability" in dataset_text
    assert "finite within-species estimate" in cross_species_text
    assert dataset_text != cross_species_text
