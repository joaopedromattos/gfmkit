from __future__ import annotations

LINK1_DATASETS: tuple[str, ...] = (
    "products_tech",
    "yelp2018",
    "yelp_textfeat",
    "products_home",
    "steam_textfeat",
    "amazon_textfeat",
    "amazon-book",
    "citation-2019",
    "citation-classic",
    "pubmed",
    "citeseer",
    "ppa",
    "p2p-Gnutella06",
    "soc-Epinions1",
    "email-Enron",
)

LINK2_DATASETS: tuple[str, ...] = (
    "Photo",
    "Goodreads",
    "Fitness",
    "ml1m",
    "ml10m",
    "gowalla",
    "arxiv",
    "arxiv-ta",
    "cora",
    "CS",
    "collab",
    "proteins_spec0",
    "proteins_spec1",
    "proteins_spec2",
    "proteins_spec3",
    "ddi",
    "web-Stanford",
    "roadNet-PA",
)

SMOKE_DATASETS: tuple[str, ...] = ("cora",)

ALL_DATASETS: tuple[str, ...] = tuple(dict.fromkeys((*LINK1_DATASETS, *LINK2_DATASETS)))

DATASET_PRESETS: dict[str, tuple[str, ...]] = {
    "all": ALL_DATASETS,
    "link1": LINK1_DATASETS,
    "link2": LINK2_DATASETS,
    "smoke": SMOKE_DATASETS,
}

