from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

def filter_by_pci(df: "pd.DataFrame", pci: int) -> "pd.DataFrame":
    return df[df["pci"].apply(lambda x: any(el == pci for el in x))]


def filter_by_mnc(df: "pd.DataFrame", mnc: int) -> "pd.DataFrame":
    return df[
        df["mcc-mnc"].apply(lambda x: any(int(el.split("-")[1]) == mnc for el in x))
    ]


def filter_by_earfcn(df: "pd.DataFrame", earfcn: int) -> "pd.DataFrame":
    return df[
        df["lte"].apply(lambda x: any(el["identity"]["earfcn"] == earfcn for el in x))
    ]
