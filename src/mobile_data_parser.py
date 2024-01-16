import pandas as pd

from src.helper import expand_dataframe, explode_nw_data
import logging as log

class MobileDataParser:
    def __init__(self, raw_data: pd.DataFrame) -> None:
        log.info("Created MobileDataParser, parsing raw data...")
        self.raw_data = self._prune_invalid_data(raw_data)
        self.measurement_points = self._parse_to_measurement_points(self.raw_data)


    def _prune_invalid_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        log.info("Pruning invalid data...")
        pruned_data = raw_data[raw_data["nw data"].apply(lambda x: len(str(x)) > 1)]
        return pruned_data

    def _parse_to_measurement_points(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        log.info("Parsing data to measurement points...")
        measurements = explode_nw_data(raw_data)
        measurements = expand_dataframe(measurements)

        # remove rows with cells which are not lte
        measurements = measurements[
            measurements["plmn lte"].apply(lambda x: len(x) > 0)
        ]

        # extract lte data from 'nw data' and convert str to correct types
        measurements["lte"] = measurements["nw data"].apply(lambda x: x.lte())
        measurements["lte"] = measurements["lte"].apply(self._to_dict)
        measurements["lte"] = measurements["lte"].apply(self._convert_types)
        measurements["time"] = measurements["time"].apply(pd.to_datetime)
        measurements[["longitude", "latitude"]] = measurements[
            ["longitude", "latitude"]
        ].apply(pd.to_numeric)

        # remove unnecessary columns
        measurements.drop("nw data", axis=1, inplace=True)
        measurements.drop("plmn wcdma", axis=1, inplace=True)

        # rename 'plmn lte' to 'mcc_mnc'
        measurements.rename(columns={"plmn lte": "mcc-mnc"}, inplace=True)
        return measurements

    def _to_dict(self, lines) -> list[dict[str, str]]:
        def _convert_to_dict(raw_cell_info: str) -> dict[str, str]:
            retval: dict[str, str] = {}
            for pair in raw_cell_info:
                pair = pair.split("=")
                if len(pair) > 1:
                    retval[pair[0]] = pair[1]
            return retval

        modified_lines = []
        for line in lines:
            line.pop("type")

            line["signal"] = _convert_to_dict(line["signal"].split()[1:])
            line["identity"] = _convert_to_dict(line["identity"].split()[1:])

            modified_lines.append(line)

        return modified_lines

    def _convert_types(self, cell_infos: list):
        signal_columns = [
            "rsrp",
            "rsrq",
            "rssi",
            "rssnr",
            "ta",
            "cqi",
            "level",
            "parametersUseForLevel",
            "cqiTableIndex",
            "miuiLevel",
            "mOptimizedLevel",
            "parametersUseForLevel"
        ]
        identity_columns = [
            "mCi",
            "mPci",
            "mTac",
            "mEarfcn",
            "mBandwidth",
            "mMcc",
            "mMnc",
            # other convention
            "pci",
            "tac",
            "earfcn",
            "bw",
            "rsrp",
            "rsrq",
            "rssi",
            "dbm",
            "ta"
        ]
        for info in cell_infos:
            for column in signal_columns:
                try:
                    info["signal"][column] = pd.to_numeric(info["signal"][column], errors="coerce", downcast="integer")
                    # if info["signal"][column] > 2147483640:
                    #     info["signal"][column] = float("inf")
                except KeyError:
                    pass
                except:
                    pass

            for column in identity_columns:
                try:
                    info["identity"][column] = int(info["identity"][column])
                except KeyError:
                    pass
                except ValueError:
                    info["identity"][column] = None
                except:
                    pass

        return cell_infos
